import os
import h5py
import torch
import gc
from pathlib import Path
from transformers import AutoModel
from tqdm import tqdm
import psutil
import time


def get_memory_usage():
    """Get current system memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # GB


def process_large_slide_in_chunks(model, features, coords, patch_size_lv0, device, chunk_size=2000):
    """
    Process very large slides by breaking them into manageable chunks
    and then aggregating the results appropriately.
    """
    n_patches = features.shape[0]
    
    if n_patches <= chunk_size:
        # Small enough to process normally
        with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
            with torch.inference_mode():
                features_gpu = features.to(device, non_blocking=True)
                coords_gpu = coords.to(device, non_blocking=True)
                
                slide_embedding = model.encode_slide_from_patch_features(features_gpu, coords_gpu, patch_size_lv0)
                
                del features_gpu, coords_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
                return slide_embedding.detach().cpu()
    
    else:
        # Process in chunks and aggregate
        print(f"  Processing {n_patches} patches in chunks of {chunk_size}...")
        
        # Strategy: Process patches in chunks, then use TITAN's aggregation method
        all_patch_features = []
        
        for start_idx in range(0, n_patches, chunk_size):
            end_idx = min(start_idx + chunk_size, n_patches)
            
            # Load chunk
            features_chunk = features[start_idx:end_idx]
            coords_chunk = coords[start_idx:end_idx]
            
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
                with torch.inference_mode():
                    features_chunk_gpu = features_chunk.to(device, non_blocking=True)
                    coords_chunk_gpu = coords_chunk.to(device, non_blocking=True)
                    
                    # Get patch-level representations (not slide-level yet)
                    # This depends on TITAN's internal structure - we'll use the full method
                    chunk_embedding = model.encode_slide_from_patch_features(features_chunk_gpu, coords_chunk_gpu, patch_size_lv0)
                    
                    all_patch_features.append(chunk_embedding.detach().cpu())
                    
                    del features_chunk_gpu, coords_chunk_gpu, chunk_embedding
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            
            # Clean up chunk data
            del features_chunk, coords_chunk
            gc.collect()
        
        # Aggregate chunk embeddings (simple approach: mean pooling)
        final_embedding = torch.stack(all_patch_features).mean(dim=0)
        
        del all_patch_features
        gc.collect()
        
        return final_embedding


def extract_slide_embeddings_from_dir(h5_dir, output_dir, pattern="*.h5", 
                                     chunk_size=2000, memory_cleanup_interval=5):
    """
    Memory-optimized TITAN slide embedding extraction that processes ALL slides.
    
    Args:
        h5_dir: Directory containing input .h5 files
        output_dir: Output directory for embeddings  
        pattern: File pattern to match
        chunk_size: Size of chunks for very large slides
        memory_cleanup_interval: Clean memory every N files
    """
    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")

    # Load TITAN model once
    print("Loading TITAN model...")
    model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    model = model.to(device)
    model.eval()

    h5_files = sorted(h5_dir.glob(pattern))
    if not h5_files:
        print(f"No files found in {h5_dir} matching pattern {pattern}")
        return

    print(f"Found {len(h5_files)} files in {h5_dir}")
    print(f"Memory after model loading: {get_memory_usage():.2f} GB")

    successes, failures = 0, 0

    for idx, h5_path in enumerate(tqdm(h5_files, desc="Extracting TITAN slide embeddings")):
        file_id = h5_path.stem
        out_pt = output_dir / f"{file_id}_slide_embedding.pt"

        # Skip if already exists
        if out_pt.exists():
            continue

        try:
            # Memory check before processing
            current_mem = get_memory_usage()
            if current_mem > 30:  # If using >30GB RAM, aggressive cleanup
                print(f"[MEMORY] Current usage: {current_mem:.2f}GB, cleaning up...")
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(2)

            # Load slide info first (just metadata)
            with h5py.File(h5_path, "r") as f:
                if "features" not in f or "coords" not in f:
                    raise KeyError(f"Missing 'features' or 'coords' in {h5_path}")

                n_patches = f["features"].shape[0]
                file_size_mb = h5_path.stat().st_size / (1024 * 1024)
                
                print(f"[INFO] {h5_path.name}: {n_patches} patches, {file_size_mb:.1f}MB")

                # Load data (all at once for small slides, or prepare for chunking)
                features = torch.from_numpy(f["features"][:])
                coords = torch.from_numpy(f["coords"][:]).long()
                
                if "patch_size_level0" not in f["coords"].attrs:
                    raise KeyError(f"coords.attrs['patch_size_level0'] missing in {h5_path}")
                patch_size_lv0 = f["coords"].attrs["patch_size_level0"]

            # Process slide (handles both small and large slides)
            slide_embedding = process_large_slide_in_chunks(
                model, features, coords, patch_size_lv0, device, chunk_size
            )

            # Save result
            torch.save(slide_embedding, out_pt)
            
            # Clean up
            del features, coords, slide_embedding
            successes += 1

            # Regular memory cleanup
            if (idx + 1) % memory_cleanup_interval == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                current_mem = get_memory_usage()
                print(f"[CLEANUP] Processed {idx+1} files, Memory: {current_mem:.2f}GB")

        except Exception as e:
            failures += 1
            print(f"[ERROR] {h5_path.name}: {e}")
            # Clean up on error
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    print(f"\nDone. Success: {successes}, Failures: {failures}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")


if __name__ == "__main__":
    extract_slide_embeddings_from_dir(
        h5_dir="/home/woody/iwi5/iwi5204h/CLAM/CONCH_1_5_features/h5_files_postprocessed",
        output_dir="/home/woody/iwi5/iwi5204h/HistGen4TITAN/slide_embeddings_new_script/",
        pattern="*.h5",
        chunk_size=2000,    # Process large slides in 2k patch chunks  
        memory_cleanup_interval=3  # Clean memory every 3 files (more frequent)
    )
