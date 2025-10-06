#!/usr/bin/env python3
"""
Script to prepare sphinx-gallery compatible structure from existing examples.
This script copies examples from the original structure to temporary gallery directories
based on configuration in gallery_config.yaml.
"""

import os
import shutil
import yaml
import glob
import json
import re
from pathlib import Path


def load_gallery_config(config_path="gallery_config.yaml"):
    """Load gallery configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_temp_galleries(galleries_config):
    """Clean existing temporary gallery directories."""
    for gallery_name, gallery_config in galleries_config['galleries'].items():
        target_dir = Path(gallery_config['target_dir'])
        if target_dir.exists():
            shutil.rmtree(target_dir)
            print(f"Cleaned {target_dir}")


def copy_examples_to_gallery(source_dir, target_dir, file_patterns, exclude_patterns, 
                           include_files=None, exclude_files=None, assets=None):
    """Copy examples from source to target directory with gallery structure."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Warning: Source directory {source_path} does not exist, skipping...")
        return []
        
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    # Get example files matching patterns
    example_files = []
    for pattern in file_patterns:
        example_files.extend(source_path.glob(pattern))
    
    # Process example files with include/exclude logic
    for file_path in example_files:
        file_name = file_path.name
        
        # Check global exclude patterns
        skip = False
        for exclude_pattern in exclude_patterns:
            if file_path.match(exclude_pattern):
                skip = True
                break
        
        if skip:
            continue
        
        # Check gallery-specific include files (if specified, only include these)
        if include_files is not None:
            include_match = False
            for include_pattern in include_files:
                if file_path.match(include_pattern):
                    include_match = True
                    break
            if not include_match:
                continue
        
        # Check gallery-specific exclude files
        if exclude_files is not None:
            exclude_match = False
            for exclude_pattern in exclude_files:
                if file_path.match(exclude_pattern):
                    exclude_match = True
                    break
            if exclude_match:
                continue
                
        # Handle different file types
        if file_path.suffix == '.ipynb':
            # Convert notebook to Python script
            target_file = target_path / (file_path.stem + '.py')
            if convert_notebook_to_python(file_path, target_file):
                copied_files.append(target_file)
        elif file_path.suffix == '.py':
            # Copy Python file directly
            target_file = target_path / file_path.name
            shutil.copy2(file_path, target_file)
            copied_files.append(target_file)
            print(f"Copied {file_path} -> {target_file}")
        else:
            # Copy data files, preserving directory structure
            rel_path = file_path.relative_to(source_path)
            target_file = target_path / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_file)
            copied_files.append(target_file)
            print(f"Copied data file {file_path} -> {target_file}")
    
    # Copy gallery-specific assets (without include/exclude filtering)
    if assets:
        for pattern in assets:
            for asset_file in source_path.rglob(pattern):
                if asset_file.is_file():
                    # Copy asset files, preserving directory structure
                    rel_path = asset_file.relative_to(source_path)
                    target_file = target_path / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(asset_file, target_file)
                    copied_files.append(target_file)
                    print(f"Copied asset {asset_file} -> {target_file}")
    
    return copied_files


def convert_notebook_to_python(notebook_path, output_path):
    """Convert Jupyter notebook to Python script."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        python_lines = []
        first_markdown_processed = False
        docstring_lines = []
        
        # Collect title and description for docstring
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'markdown' and not first_markdown_processed:
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                
                # Use first markdown cell as docstring content with RST formatting
                docstring_lines = convert_markdown_to_rst(source)
                
                first_markdown_processed = True
                break
        
        # Create docstring at the beginning
        if not docstring_lines:
            # Fallback title based on filename
            title = output_path.stem.replace('_', ' ').title()
            docstring_lines = [title, '=' * len(title), "", f"Example from {notebook_path.name}"]
        
        # Generate proper docstring
        python_lines.append('"""')
        for line in docstring_lines:
            python_lines.append(line)
        python_lines.append('"""')
        python_lines.append('')
        
        # Process remaining cells (skip first markdown cell since it's now the docstring)
        skip_first_markdown = True
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                
                # Check if this cell contains only pip magic
                if source.strip().startswith('!pip'):
                    # Convert pip magic to markdown section
                    command = source.strip()[1:].strip()
                    python_lines.append('# %%')
                    # python_lines.append('# Installation')
                    # python_lines.append('#')
                    python_lines.append('# .. code-block:: bash')
                    python_lines.append('#')
                    python_lines.append(f'#    {command}')
                    python_lines.append('#')
                    python_lines.append('')
                else:
                    # Add cell separator comment for sphinx-gallery
                    python_lines.append('# %%')
                    
                    # Handle magics
                    processed_source = process_magics(source)
                    python_lines.append(processed_source)
                    python_lines.append('')
            
            elif cell.get('cell_type') == 'markdown':
                if skip_first_markdown:
                    # Skip the first markdown cell since it's already used as docstring
                    skip_first_markdown = False
                    continue
                
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                
                # Convert markdown to comments with RST formatting
                python_lines.append('# %%')
                markdown_lines = convert_markdown_to_rst(source)
                for line in markdown_lines:
                    if line.strip():
                        python_lines.append(f'# {line}')
                    else:
                        python_lines.append('#')
                python_lines.append('')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(python_lines))
        
        print(f"Converted notebook {notebook_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting notebook {notebook_path}: {e}")
        return False


def convert_markdown_to_rst(markdown_text):
    """Convert markdown formatting to RST formatting."""
    lines = markdown_text.split('\n')
    rst_lines = []
    in_code_block = False
    code_language = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting code block
                in_code_block = True
                code_language = line.strip()[3:].strip() or 'none'
                rst_lines.append(f'.. code-block:: {code_language}')
                rst_lines.append('')
            else:
                # Ending code block
                in_code_block = False
                rst_lines.append('')
            i += 1
            continue
        
        if in_code_block:
            # Inside code block - add indentation for RST
            rst_lines.append(f'   {line}')
            i += 1
            continue
        
        # Handle headers
        if line.startswith('###'):
            title = line[3:].strip()
            rst_lines.append(title)
            rst_lines.append('-' * len(title))
        elif line.startswith('##'):
            title = line[2:].strip()
            rst_lines.append(title)
            rst_lines.append('-' * len(title))
        elif line.startswith('#'):
            title = line[1:].strip()
            rst_lines.append(title)
            rst_lines.append('=' * len(title))
        else:
            # Handle inline formatting
            rst_line = line
            
            # Convert <code>text</code> to ``text`` (HTML code tags to RST)
            rst_line = re.sub(r'<code>([^<]+)</code>', r'`\1`', rst_line)
            
            # Convert `code` to ``code`` (RST uses double backticks)
            rst_line = re.sub(r'`([^`]+)`', r'``\1``', rst_line)
            
            # Handle bold and italic formatting carefully
            # Convert __bold__ to **bold** (RST uses ** for bold)
            rst_line = re.sub(r'__([^_]+)__', r'**\1**', rst_line)
            
            # Convert _italic_ to *italic* (RST uses * for italic)
            rst_line = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'*\1*', rst_line)
            
            # Handle markdown links: [text](url) -> `text <url>`_
            rst_line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'`\1 <\2>`_', rst_line)
            
            rst_lines.append(rst_line)
        
        i += 1
    
    return rst_lines


def process_magics(source):
    """Process Jupyter magics and convert them to regular Python code."""
    lines = source.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Handle shell commands (! prefix)
        if stripped.startswith('!'):
            command = stripped[1:].strip()
            if command.startswith('pip'):
                # Pip commands are handled separately as markdown sections
                # Skip them here since they're processed at cell level
                continue
            else:
                # Convert other shell commands to subprocess calls
                processed_lines.append(f'import subprocess')
                processed_lines.append(f'subprocess.run(["{command}"], shell=True)')
        
        # Handle line magics (% prefix)
        elif stripped.startswith('%'):
            if stripped.startswith('%matplotlib'):
                processed_lines.append('import matplotlib.pyplot as plt')
            elif stripped.startswith('%time'):
                # Remove %time magic, keep the code
                code = stripped[5:].strip()
                if code:
                    processed_lines.append(code)
            elif stripped.startswith('%load_ext'):
                # Convert to comment
                processed_lines.append(f'# {line}')
            else:
                # Comment out other magics
                processed_lines.append(f'# {line}')
        
        # Handle cell magics (%% prefix)
        elif stripped.startswith('%%'):
            if stripped.startswith('%%time'):
                # Comment out cell magic, keep following code
                processed_lines.append(f'# {line}')
            else:
                # Comment out other cell magics
                processed_lines.append(f'# {line}')
        
        else:
            # Regular Python code
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def create_gallery_files(target_dir, title, description):
    """Create README.txt and GALLERY_HEADER.rst for sphinx-gallery."""
    # Create README.txt
    readme_content = f"""{title}
{'=' * len(title)}

{description}
"""
    
    readme_path = Path(target_dir) / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Created {readme_path}")
    
    # Create GALLERY_HEADER.rst
    header_content = f"""
{title}
{'=' * len(title)}

{description}
"""
    
    header_path = Path(target_dir) / "GALLERY_HEADER.rst"
    with open(header_path, 'w') as f:
        f.write(header_content)
    print(f"Created {header_path}")


def prepare_galleries():
    """Main function to prepare all galleries."""
    # Change to docs directory to ensure relative paths work
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)
    
    # Load configuration
    config = load_gallery_config()
    
    # Clean existing temp galleries
    clean_temp_galleries(config)
    
    # Prepare each gallery
    gallery_dirs = []
    
    for gallery_name, gallery_config in config['galleries'].items():
        print(f"\nPreparing gallery: {gallery_name}")
        
        source_dir = gallery_config['source_dir']
        target_dir = gallery_config['target_dir']
        title = gallery_config['title']
        description = gallery_config['description']
        
        # Get gallery-specific include/exclude options
        include_files = gallery_config.get('include_files')
        exclude_files = gallery_config.get('exclude_files')
        assets = gallery_config.get('assets')
        
        # Copy examples
        copied_files = copy_examples_to_gallery(
            source_dir, 
            target_dir,
            config['file_patterns'],
            config['exclude_patterns'],
            include_files,
            exclude_files,
            assets
        )
        
        if copied_files:
            # Create README.txt and GALLERY_HEADER.rst for gallery
            create_gallery_files(target_dir, title, description)
            gallery_dirs.append(target_dir)
        else:
            print(f"No files copied for gallery {gallery_name}")
    
    print(f"\nPrepared {len(gallery_dirs)} galleries:")
    for gallery_dir in gallery_dirs:
        print(f"  - {gallery_dir}")
    
    return gallery_dirs


if __name__ == "__main__":
    prepare_galleries()