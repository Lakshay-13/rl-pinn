import subprocess
import time
import pandas as pd
import psutil
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def normalize_device_arg(device):
    device_str = str(device)
    if device_str.startswith("cuda"):
        return "cuda"
    if device_str.startswith("mps"):
        return "mps"
    if device_str.startswith("cpu"):
        return "cpu"
    return device_str


def get_peak_memory(cmd):
    """Runs command, monitors peak RSS memory usage, and returns exit details."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_mem = 0
    try:
        ps_proc = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                current_mem = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    current_mem += child.memory_info().rss
                peak_mem = max(peak_mem, current_mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)
    except Exception:
        process.kill()
        raise

    stdout, stderr = process.communicate()
    return peak_mem, process.returncode, stdout, stderr

def perform_scan(cinns_path=".", mem_limit=0.85, device="auto", target_memories=50000, steps=833):
    """
    Scans hardware to find optimal worker count.
    Tests W workers with W episodes (1 ep per worker).
    """
    run_py = str(SCRIPT_DIR / "run.py")
    cinns_path = str(Path(cinns_path).resolve())
    device = normalize_device_arg(device)

    workers_list = [1, 2, 4, 8, 16, 32, 48, 64, 96]
    if device == "cpu":
        cpu_count = max(1, psutil.cpu_count(logical=True) or 1)
        workers_list = [w for w in workers_list if w <= cpu_count]
        if cpu_count not in workers_list:
            workers_list.append(cpu_count)
        workers_list = sorted(set(workers_list))

    results = []
    mem_per_worker = 0
    available_ram = psutil.virtual_memory().available
    
    print(f"\n--- Starting Hardware Scan on {device.upper()} (Limit: {mem_limit*100}% Available RAM) ---")
    print(f"Initial Available RAM: {available_ram / 1024**3:.2f} GB")

    for w in workers_list:
        # Memory Guard
        if mem_per_worker > 0:
            estimated_mem = mem_per_worker * w
            if estimated_mem > available_ram * mem_limit:
                print(f"[SCAN] Stopping at {w} workers: Next step would exceed memory safety limit.")
                break

        # Run 1 episode per worker for a quick test
        # (w workers, w total episodes = 1 ep per worker)
        episodes = w 
        
        cmd = [
            sys.executable, run_py,
            "--device", device,
            "--workers", str(w),
            "--cycles", "1",
            "--target_memories", str(episodes * steps),
            "--train_epochs", "1",
            "--log_dir", "logs/scan_tmp",
            "--cinns_path", cinns_path,
            "--skip_render",
            "--skip_device_scan",
        ]

        start_time = time.time()
        try:
            if w == 1:
                peak, returncode, _, stderr = get_peak_memory(cmd)
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)
                mem_per_worker = peak * 1.15 # 15% safety buffer
                elapsed = time.time() - start_time
            else:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                elapsed = time.time() - start_time
            
            # ep_per_min = (total episodes / total time) * 60
            ep_per_min = (episodes / elapsed) * 60
            results.append({"Workers": w, "Ep/Min": ep_per_min})
            print(f"Workers: {w} | Throughput: {ep_per_min:.2f} Ep/Min")
            
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            print(f"[SCAN] Error testing {w} workers: {e}")
            if stderr:
                print("[SCAN] Child stderr:")
                print(stderr[-1000:])
            break
        except Exception as e:
            print(f"[SCAN] Error testing {w} workers: {e}")
            break

    if not results:
        return 1, 20 # Fallback

    # Determine Optimal (Elbow Method)
    df = pd.DataFrame(results)
    if len(df) > 1:
        df['Worker_Increase'] = df['Workers'].pct_change()
        df['Throughput_Increase'] = df['Ep/Min'].pct_change()
        df['Efficiency'] = df['Throughput_Increase'] / df['Worker_Increase']
        
        # Optimal is the highest count with > 50% efficiency gain
        threshold = 0.5
        df_valid = df.fillna({'Efficiency': 1.0})
        optimal_row = df_valid[df_valid['Efficiency'] >= threshold].iloc[-1:]
        optimal_w = int(optimal_row['Workers'].values[0])
    else:
        optimal_w = int(df['Workers'].iloc[0])

    # Recommend a worker-aligned episode count that stays close to the
    # requested cycle memory budget.
    desired_episodes = max(1, round(target_memories / steps))
    recommended_eps = max(optimal_w, round(desired_episodes / optimal_w) * optimal_w)
    
    print(f"\n--- Scan Result ---")
    print(f"Optimal Workers: {optimal_w}")
    print(f"Recommended Episodes per Cycle: {recommended_eps} (~{recommended_eps * steps} memories)")
    
    return optimal_w, recommended_eps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cinns_path", type=str, default=None)
    parser.add_argument("--mem_limit", type=float, default=0.85)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--target_memories", type=int, default=50000)
    parser.add_argument("--steps", type=int, default=833)
    args = parser.parse_args()
    
    if args.cinns_path is None:
        args.cinns_path = str(SCRIPT_DIR)
        
    perform_scan(args.cinns_path, args.mem_limit, args.device, args.target_memories, args.steps)
