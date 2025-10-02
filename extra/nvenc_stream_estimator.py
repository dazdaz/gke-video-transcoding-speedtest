#!/usr/bin/env python3

import argparse
import math

def estimate_nvenc_streams(
    gpu_input_name: str, # Renamed to avoid confusion with internal gpu_architecture variable
    codec_type: str,
    preset: str,
    rc_mode: str,
    desired_fps_per_stream: int,
    num_nvenc_units: int = None
) -> int:
    """
    Estimates the maximum number of simultaneous video streams that can be
    encoded by a given NVIDIA GPU (based on architecture) using NVENC,
    leveraging provided benchmark data.

    Args:
        gpu_input_name (str): The NVIDIA GPU architecture (e.g., "Pascal", "Turing", "Ampere", "Ada", "Blackwell", "Hopper")
                                or a common GPU model (e.g., "T4", "L4", "RTX 4090", "H100").
        codec_type (str): The video codec type ("H.264" or "HEVC").
        preset (str): The encoding preset (e.g., "p1", "p3", "p5", "p7").
        rc_mode (str): The rate control mode ("CBR" for Constant Bitrate, "VBR" for Variable Bitrate).
        desired_fps_per_stream (int): The target frames per second for each individual stream.
        num_nvenc_units (int, optional): The number of NVENC units on the specific GPU model.
                                          If provided, the total estimated FPS will be scaled based on
                                          the typical NVENC unit count for the architecture.
                                          If None, the script assumes the typical NVENC unit count for
                                          the architecture (e.g., Ada is typically 2, but L4 has 1; Hopper typically 3).

    Returns:
        int: The estimated maximum number of simultaneous streams.
             Returns -1 if the specified configuration is not found in the data or if inputs are invalid.
    """

    # Data extracted from the provided image table.
    # IMPORTANT: Hopper (H100) data is NOT available in the original benchmark table.
    # The "Hopper" entries below are **speculative placeholders**.
    # For this script, we are roughly using "Blackwell" data as a proxy for Hopper's NVENC performance,
    # and adjusting based on the known number of NVENC units for H100 (3 units).
    # Actual Hopper NVENC performance may differ significantly.
    encoding_data = {
        "p1": {
            "CBR": {
                "H.264": {"Pascal": 667, "Turing": 855, "Ampere": 868, "Ada": 910, "Hopper": 990, "Blackwell": 977},
                "HEVC":  {"Pascal": 539, "Turing": 932, "Ampere": 943, "Ada": 1055, "Hopper": 1150, "Blackwell": 1134}
            },
            "VBR": {
                "H.264": {"Pascal": 692, "Turing": 833, "Ampere": 846, "Ada": 885, "Hopper": 960, "Blackwell": 948},
                "HEVC":  {"Pascal": 506, "Turing": 920, "Ampere": 939, "Ada": 1037, "Hopper": 1130, "Blackwell": 1119}
            }
        },
        "p3": {
            "CBR": {
                "H.264": {"Pascal": 649, "Turing": 600, "Ampere": 613, "Ada": 652, "Hopper": 730, "Blackwell": 718},
                "HEVC":  {"Pascal": 442, "Turing": 463, "Ampere": 467, "Ada": 494, "Hopper": 540, "Blackwell": 529}
            },
            "VBR": {
                "H.264": {"Pascal": 398, "Turing": 602, "Ampere": 617, "Ada": 647, "Hopper": 720, "Blackwell": 708},
                "HEVC":  {"Pascal": 443, "Turing": 552, "Ampere": 557, "Ada": 706, "Hopper": 960, "Blackwell": 947}
            }
        },
        "p5": {
            "CBR": {
                "H.264": {"Pascal": 363, "Turing": 271, "Ampere": 273, "Ada": 291, "Hopper": 330, "Blackwell": 323},
                "HEVC":  {"Pascal": 370, "Turing": 305, "Ampere": 307, "Ada": 343, "Hopper": 520, "Blackwell": 506}
            },
            "VBR": {
                "H.264": {"Pascal": 327, "Turing": 264, "Ampere": 266, "Ada": 283, "Hopper": 320, "Blackwell": 317},
                "HEVC":  {"Pascal": 371, "Turing": 334, "Ampere": 335, "Ada": 411, "Hopper": 530, "Blackwell": 521}
            }
        },
        "p7": {
            "CBR": {
                "H.264": {"Pascal": 321, "Turing": 229, "Ampere": 231, "Ada": 247, "Hopper": 270, "Blackwell": 264},
                "HEVC":  {"Pascal": 345, "Turing": 306, "Ampere": 308, "Ada": 343, "Hopper": 470, "Blackwell": 464}
            },
            "VBR": {
                "H.264": {"Pascal": 250, "Turing": 207, "Ampere": 213, "Ada": 211, "Hopper": 230, "Blackwell": 227},
                "HEVC":  {"Pascal": 260, "Turing": 171, "Ampere": 171, "Ada": 181, "Hopper": 190, "Blackwell": 181}
            }
        }
    }
    # These "Hopper" values are very rough estimates, slightly adjusted from "Blackwell"
    # to account for potential improvements or simply to be distinct.
    # REAL BENCHMARK DATA FOR H100 IS REQUIRED FOR ACCURATE RESULTS.


    # Map common GPU models to their architecture
    gpu_model_to_architecture = {
        "T4": "Turing",
        "L4": "Ada", # NVIDIA L4 uses Ada Lovelace architecture
        "H100": "Hopper", # NVIDIA H100 uses Hopper architecture
        "RTX 3090": "Ampere",
        "RTX 4090": "Ada",
        "GTX 1060": "Pascal",
        # Add more mappings as needed
    }

    # Typical NVENC units for architectures as likely represented in the benchmark table
    # or commonly known for server/data center GPUs.
    # This is used for scaling if `num_nvenc_units` is provided by the user for a specific card.
    # These reflect the *benchmark card's* unit count.
    typical_nvenc_units_for_benchmark = {
        "Pascal": 1,   # e.g., GTX 1060, P100, P4
        "Turing": 2,   # e.g., RTX 20 series, RTX 8000. T4 is an outlier with 1.
        "Ampere": 2,   # e.g., RTX 30 series, A100
        "Ada": 2,      # e.g., RTX 40 series. L4 is an outlier with 1, L40/L40S have 3.
        "Hopper": 3,   # H100 has 3 NVENC units.
        "Blackwell": 3 # Assumption, likely 3 or more in high-end models.
    }

    # Normalize inputs and resolve architecture from model name
    # We create a new variable `gpu_architecture` for the resolved architecture name.
    # The original `gpu_input_name` is used for printing the user's input.
    gpu_architecture = gpu_input_name.capitalize()
    if gpu_architecture in gpu_model_to_architecture:
        gpu_architecture = gpu_model_to_architecture[gpu_architecture]

    codec_type = codec_type.upper()
    preset = preset.lower()
    rc_mode = rc_mode.upper()

    # Now print with the original input and the resolved architecture
    print(f"\n--- Estimating Streams for {gpu_input_name} ({gpu_architecture} Architecture) ---")
    print(f"Codec: {codec_type}, Preset: {preset}, RC Mode: {rc_mode}")
    print(f"Desired FPS per Stream: {desired_fps_per_stream} fps")

    # 1. Validate inputs and retrieve EncoderMaxFPS_SingleStream (from benchmarked typical config)
    preset_data = encoding_data.get(preset)
    if not preset_data:
        print(f"Error: Invalid preset '{preset}'. Available presets: {list(encoding_data.keys())}")
        return -1

    rc_mode_data = preset_data.get(rc_mode)
    if not rc_mode_data:
        print(f"Error: Invalid RC Mode '{rc_mode}'. Available RC modes: {list(preset_data.keys())}")
        return -1

    codec_data = rc_mode_data.get(codec_type)
    if not codec_data:
        print(f"Error: Invalid Codec Type '{codec_type}'. Available codecs: {list(rc_mode_data.keys())}")
        return -1

    # This is where the lookup happens. It now uses the resolved `gpu_architecture` (e.g., "Ada" for "L4")
    encoder_max_fps_benchmarked = codec_data.get(gpu_architecture)
    if encoder_max_fps_benchmarked is None:
        print(f"Error: No benchmark data for GPU architecture '{gpu_architecture}' under {codec_type}/{preset}/{rc_mode}. Available architectures: {list(codec_data.keys())}")
        return -1

    if desired_fps_per_stream <= 0:
        print("Error: Desired FPS per stream must be a positive integer.")
        return -1

    # --- WARNING FOR HOPPER DATA ---
    if gpu_architecture == "Hopper":
        print("\n!!! WARNING: Hopper (H100) NVENC benchmark data is NOT officially available in this script's dataset. !!!")
        print("!!! The performance numbers used for Hopper are **SPECULATIVE PLACEHOLDERS** (based on Blackwell data) !!!")
        print("!!! and may not reflect actual H100 NVENC performance. Use with caution. !!!\n")
        # For a truly accurate estimate, real H100 NVENC benchmark data is required.
    # --- END WARNING ---

    encoder_max_fps_benchmarked = float(encoder_max_fps_benchmarked)
    total_encoder_fps = encoder_max_fps_benchmarked

    # Apply scaling based on user-provided NVENC units vs. benchmarked units
    if num_nvenc_units is not None:
        print(f"Number of NVENC units specified: {num_nvenc_units}")
        benchmarked_units = typical_nvenc_units_for_benchmark.get(gpu_architecture)

        if benchmarked_units is None:
            print(f"Warning: Could not determine typical NVENC units for benchmarked '{gpu_architecture}'. Cannot accurately scale based on provided `num_nvenc_units`.")
            # In this case, we proceed with the raw benchmarked FPS as total
        elif num_nvenc_units == benchmarked_units:
            print(f"Provided NVENC units ({num_nvenc_units}) match typical units for benchmarked {gpu_architecture}.")
            # No scaling needed as it matches the benchmark's presumed config
        else:
            print(f"Adjusting FPS: Benchmarked {gpu_architecture} assumed to have {benchmarked_units} NVENCs. Your GPU has {num_nvenc_units} NVENCs.")
            # Calculate per-unit FPS from benchmark, then scale by user's units
            fps_per_nvenc_unit = encoder_max_fps_benchmarked / benchmarked_units
            total_encoder_fps = fps_per_nvenc_unit * num_nvenc_units
            print(f"Adjusted Total Encoder FPS: {total_encoder_fps:.2f} fps")
    else:
        # Default to typical NVENC units if not specified by user
        default_units = typical_nvenc_units_for_benchmark.get(gpu_architecture, 'unknown')
        print(f"Using aggregate performance from benchmark table for {gpu_architecture} (assuming {default_units} NVENCs).")
        print("For specific cards like L4 (1 NVENC) or T4 (1 NVENC), or L40/L40S (3 NVENCs), consider providing '--nvenc_units X' for more accurate results.")


    print(f"Effective Total Encoder FPS Capacity: {total_encoder_fps:.2f} fps")


    # 2. Calculate the Estimated Number of Streams
    estimated_num_streams = total_encoder_fps / desired_fps_per_stream
    print(f"Estimated (fractional) Streams: {estimated_num_streams:.2f}")

    # 3. Round Down to the Nearest Whole Number
    max_simultaneous_streams = math.floor(estimated_num_streams)
    print(f"Max Simultaneous Streams (rounded down): {max_simultaneous_streams}")

    return max_simultaneous_streams

def main():
    parser = argparse.ArgumentParser(
        description="Estimate the maximum number of simultaneous video streams for NVIDIA GPUs using NVENC.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "gpu",
        type=str,
        help="NVIDIA GPU architecture or common model (e.g., Pascal, Turing, Ampere, Ada, Hopper, Blackwell, T4, L4, H100).\n"
             "Note: T4 is mapped to Turing, L4 is mapped to Ada, H100 is mapped to Hopper."
    )
    parser.add_argument(
        "codec",
        type=str,
        choices=["H.264", "HEVC"],
        help="Video codec type (H.264 or HEVC)."
    )
    parser.add_argument(
        "preset",
        type=str,
        choices=["p1", "p3", "p5", "p7"],
        help="Encoding preset (p1, p3, p5, p7). Lower numbers often mean higher quality/lower speed."
    )
    parser.add_argument(
        "rc_mode",
        type=str,
        choices=["CBR", "VBR"],
        help="Rate control mode (CBR for Constant Bitrate, VBR for Variable Bitrate)."
    )
    parser.add_argument(
        "fps_per_stream",
        type=int,
        help="Target frames per second for each individual output stream (e.g., 30, 60)."
    )
    parser.add_argument(
        "--nvenc_units",
        type=int,
        help="Optional: Number of NVENC units on your specific GPU card. "
             "If provided, the estimate will be scaled based on this. "
             "E.g., T4 has 1 NVENC unit, L4 has 1 NVENC unit, L40/L40S have 3, H100 has 3, many RTX cards have 2.\n"
             "Providing this value can improve accuracy for cards with unit counts that differ from the "
             "architecture's typical benchmarked configuration.",
        default=None
    )

    args = parser.parse_args()

    result = estimate_nvenc_streams(
        gpu_input_name=args.gpu, # Pass the original GPU input name
        codec_type=args.codec,
        preset=args.preset,
        rc_mode=args.rc_mode,
        desired_fps_per_stream=args.fps_per_stream,
        num_nvenc_units=args.nvenc_units
    )

    if result != -1:
        print(f"\nConclusion: Your {args.gpu} can theoretically handle approximately {result} simultaneous {args.fps_per_stream}fps {args.codec} streams.")
    else:
        print("\nEstimation failed. Please check the provided arguments and data.")

    print("\n--- Important Considerations ---")
    print("- This estimate is based on peak single-stream performance and may not reflect real-world multi-stream scenarios due to:")
    print("  - System overhead (CPU, memory bandwidth, PCIe bandwidth).")
    print("  - Desired quality vs. speed trade-offs (presets).")
    print("  - Input video characteristics (resolution, format, content complexity).")
    print("- The table values are for 1920x1080 YUV 4:2:0/8-bit content.")
    print("- NVDEC (decoding) capacity is separate and also has limits. This script only estimates NVENC (encoding).")
    print("- The interpretation of 'nvenc_units' for scaling is an approximation based on available benchmark details. Actual performance may vary.")
    print("- **ATTENTION: Hopper (H100) performance data is speculative. For accurate H100 results, find official NVIDIA NVENC benchmarks.**")

if __name__ == "__main__":
    main()
