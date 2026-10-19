import json


def job_name(job):
    if "name" in job:
        return job["name"]
    return "/".join(str(job[key]) for key in ["arm", "threshold", "seed", "fold", "condition"])


def authorized_jobs(root, matrix, jobs):
    policy = root / "runs/evaluation_scope.json"
    if not policy.exists():
        return jobs, []
    information = json.loads(policy.read_text())
    assert information["esm_msr"]["further_training_allowed"] is False
    source = json.loads((root / information["esm_msr"]["source_manifest"]).read_text())
    assert source["checkpoint_ready"] and source["best_epoch"] == 9
    names = information["matrices"][matrix]["discontinued_job_names"]
    assert len(names) == len(set(names))
    assert set(names) <= {job_name(job) for job in jobs}
    excluded = set(names)
    return [job for job in jobs if job_name(job) not in excluded], names


def scope_status(total, discontinued, completed):
    expected = total - len(discontinued)
    return {"expected_evaluations": total, "authorized_evaluations": expected,
            "discontinued_evaluations": len(discontinued),
            "scope_policy": "runs/evaluation_scope.json" if discontinued else None,
            "active_scope_complete": completed == expected,
            "complete": completed == total}
