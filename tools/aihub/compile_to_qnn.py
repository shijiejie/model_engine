"""Compile an ONNX/PyTorch model to a QNN context binary via Qualcomm AI Hub.

No local QAIRT toolchain needed. Picks the first SM8750 (Snapdragon 8 Elite,
Hexagon V79) device in the pool — same SoC family as the test phone.

The API token must come from the environment (see doc/readme_qnn.md for how to
get one) — it is never written to disk by this script.

Usage:
    python tools/aihub/compile_to_qnn.py <model.onnx|model.pt2> [out.bin]
"""
import os
import sys

import qai_hub as hub


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: compile_to_qnn.py <model.onnx|model.pt2> [out.bin]")
    model_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "model_aihub.bin"

    token = os.environ.get("QAI_HUB_TOKEN")
    if not token:
        sys.exit("QAI_HUB_TOKEN not set (see doc/readme_qnn.md)")
    hub.set_session_token(token)  # 0.56 API; deprecated but functional

    device = next(d for d in hub.get_devices()
                  if any("8750" in a for a in d.attributes))
    print("device:", device.name)

    # Returns (list[CompileJob], LinkJob); the link job's target model IS the
    # raw context binary.
    cjobs, link_job = hub.submit_compile_and_link_jobs(
        models=model_path,
        device=device,
    )
    link_job.wait()
    status = link_job.get_status()
    print("link job status:", status)
    if "SUCCESS" not in str(status):
        cjobs[0].download_job_logs("compile.log")  # why it failed
        sys.exit("compile failed; see compile.log")

    model = link_job.get_target_model()
    model.download(out_path)
    print(f"downloaded {out_path}, {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
