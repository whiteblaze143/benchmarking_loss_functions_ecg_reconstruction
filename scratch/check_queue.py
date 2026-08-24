import json

try:
    with open("refine-logs/queue/queue_state.json") as f:
        data = json.load(f)
    
    queue = data.get("jobs", [])
    completed = []
    running = []
    failed = []
    retry_queued = []
    pending = []
    
    for job_info in queue:
        job_id = job_info.get("id", job_info.get("job_id", "unknown"))
        status = job_info.get("status")
        if status == "completed":
            completed.append(job_id)
        elif status == "running":
            running.append(job_id)
        elif status in ["failed_oom", "failed_transient"] or (
            status == "pending"
            and job_info.get("attempts", 0) > 0
            and job_info.get("error")
        ):
            retry_queued.append(job_id)
        elif status in ["failed", "stuck", "failed_other", "CUDA OOM detected"]:
            failed.append(job_id)
        else:
            pending.append(job_id)
            
    print(f"Total jobs: {len(queue)}")
    print(f"Completed: {len(completed)}")
    print(f"Running: {len(running)}")
    print(f"Failed/Stuck: {len(failed)}")
    print(f"Retry queued: {len(retry_queued)}")
    print(f"Pending: {len(pending)}")
    
    if failed:
        print("\nSome failed jobs:")
        for job_info in [j for j in queue if j.get("status") in ["failed", "stuck", "failed_other", "CUDA OOM detected"]][:5]:
            print(
                f"  {job_info.get('id', job_info.get('job_id', 'unknown'))}: "
                f"{job_info.get('error', 'Unknown error')}"
            )

    if retry_queued:
        print("\nRetry-queued jobs:")
        for job_info in [
            j for j in queue
            if j.get("id", j.get("job_id")) in retry_queued
        ][:5]:
            print(
                f"  {job_info.get('id', job_info.get('job_id', 'unknown'))}: "
                f"attempts={job_info.get('attempts', 0)}, "
                f"reason={job_info.get('error', 'Unknown error')}"
            )
            
except Exception as e:
    print(f"Error: {e}")
