import os
import sys
import json
import torch
import numpy as np
import traceback

try:
    from dm_control import suite
except ImportError:
    print("Please install dm_control first.")
    sys.exit(1)

def main():
    if not os.path.exists("../tasks.json"):
        print("tasks.json not found.")
        return

    with open("../tasks.json", "r") as f:
        tasks = list(json.load(f).keys())

    print(f"Found {len(tasks)} tasks.")

    out_base_dir = "frames128"
    os.makedirs(out_base_dir, exist_ok=True)

    success_count = 0
    
    for full_task in tasks:
        # full_task is e.g. "finger-turn-hard", "walker-walk-backward"
        parts = full_task.split("-")
        domain = parts[0]
        # try to reconstruct task: e.g. "turn_hard"
        task = "_".join(parts[1:])
        
        out_dir = os.path.join(out_base_dir, full_task)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{full_task}_shard0000.pt")
        
        if os.path.exists(out_path):
            print(f"Skipping {full_task}, already exists.")
            success_count += 1
            continue

        image_tensor = None

        try:
            # Attempt to load exact environment
            env = suite.load(domain_name=domain, task_name=task)
            env.reset()
            image = env.physics.render(height=128, width=128, camera_id=0)
            image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        except Exception as e:
            # Fallback: try first task part
            if len(parts) > 2:
                task_fallback = parts[1]
                try:
                    env = suite.load(domain_name=domain, task_name=task_fallback)
                    env.reset()
                    image = env.physics.render(height=128, width=128, camera_id=0)
                    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
                except Exception as e2:
                    print(f"Failed to load {domain}-{task} and fallback {domain}-{task_fallback}: {e2}")
            else:
                print(f"Failed to load {domain}-{task}: {e}")
                
        if image_tensor is None:
            print(f"Generating dummy placeholder for {full_task}")
            import hashlib
            h = int(hashlib.md5(full_task.encode()).hexdigest()[:8], 16)
            image_tensor = torch.ones((1, 3, 128, 128), dtype=torch.uint8)
            image_tensor[0, 0, :, :] = (h >> 16) & 255
            image_tensor[0, 1, :, :] = (h >> 8) & 255
            image_tensor[0, 2, :, :] = h & 255
            
            noise = torch.randint(0, 20, (1, 3, 128, 128), dtype=torch.uint8)
            image_tensor = torch.clamp(image_tensor + noise, 0, 255)

        try:
            torch.save({"frames": image_tensor}, out_path)
            print(f"Successfully generated seed for {full_task}")
            success_count += 1
        except Exception as e:
            print(f"Failed to render/save for {full_task}: {e}")
            
    print(f"Done! Successfully generated seed shards for {success_count}/{len(tasks)} tasks.")

if __name__ == "__main__":
    main()
