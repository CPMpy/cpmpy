from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from cpmpy.tools.benchmark.test.manager import RunExecResourceManager, run_instance
from cpmpy.tools.benchmark.test.xcsp3_instance_runner import XCSP3InstanceRunner


def worker_function(worker_id, cores, job_queue, time_limit, memory_limit):
    """Worker function that picks jobs from the queue until it's empty."""
    # Recreate instances in each worker process (they may not be picklable)
    resource_manager = RunExecResourceManager()
    instance_runner = XCSP3InstanceRunner()
    
    while True:
        try:
            # Get a job from the queue (blocks until one is available)
            instance, metadata = job_queue.get_nowait()
        except Exception:
            # Queue is empty, worker is done
            break
        
        # Run the instance with this worker's assigned cores
        run_instance(instance, instance_runner, time_limit, memory_limit, cores, resource_manager)
        job_queue.task_done()


def main():
    from cpmpy.tools.dataset.model.xcsp3 import XCSP3Dataset

    # dataset = XCSP3Dataset(root="./data", year=2025, track="CSP25", download=True)
    # dataset = OPBDataset(root="./data", year=2024, track="DEC-LIN", download=True)
    # dataset = JSPLibDataset(root="./data", download=True)
    dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)

    time_limit = 10*60
    workers = 1
    cores_per_worker = 1
    total_memory = 25000
    memory_per_worker = total_memory // workers
    memory_limit = memory_per_worker# Bytes to MB
    # resource_manager = RunExecResourceManager()
    # instance_runner = XCSP3InstanceRunner()

    # Calculate core assignments for each worker
    # Each worker gets a fixed set of consecutive cores
    import psutil
    total_cores = psutil.cpu_count(logical=False)  # physical cores
    # total_cores = psutil.cpu_count(logical=True)  # logical cores (with hyperthreading)
    
    if workers * cores_per_worker > total_cores:
        raise ValueError(f"Not enough cores: {workers} workers × {cores_per_worker} cores = {workers * cores_per_worker} cores needed, but only {total_cores} available")
    
    # Assign cores to each worker
    worker_cores = []
    for i in range(workers):
        start_core = i * cores_per_worker
        end_core = start_core + cores_per_worker
        cores = list(range(start_core, end_core))
        worker_cores.append(cores)
    
    print(f"Total cores: {total_cores}, Workers: {workers}, Cores per worker: {cores_per_worker}")
    for i, cores in enumerate(worker_cores):
        print(f"Worker {i}: cores {cores}")

    # Create a queue of all jobs using Manager for ProcessPoolExecutor compatibility
    with Manager() as manager:
        job_queue = manager.Queue()
        for instance, metadata in dataset:
            job_queue.put((instance, metadata))
        
        # Submit workers to the executor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(worker_function, worker_id, cores, job_queue, time_limit, memory_limit)
                for worker_id, cores in enumerate(worker_cores)
            ]
            # Wait for all workers to finish
            for future in futures:
                future.result()

        

if __name__ == "__main__":
    main()