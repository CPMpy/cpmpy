

Example command to run:
```python
python cpmpy/tools/xcsp3/xcsp3_cpmpy.py cpmpy/tools/xcsp3/models/Fillomino-mini-5-0_c24.xml
```


The runner currently makes use of the `resource` module of python (for setting a limit on memory consumption), which is UNIX exclusive. Windows seems to require more manual fiddeling with the win32 API: https://stackoverflow.com/questions/54949110/limit-python-script-ram-usage-in-windows