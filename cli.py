"""
Resource-Aware Embedded Audio Processing and Intelligent Source Separation System
Headless CLI Execution Interface (Linux / Raspberry Pi / Headless Terminal)
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.audio_input import FileAudioSource
from core.separation_engine import UNetSourceSeparator, DummySourceSeparator
from core.resource_controller import ResourceController, ProcessingMode


def main():
    parser = argparse.ArgumentParser(
        description="Resource-Aware Embedded Audio Processing and Source Separation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-a", "--audio", type=str, default="sample_mixture.wav", help="Input audio path (WAV/MP3/FLAC)")
    parser.add_argument("-m", "--mode", type=str, default="BALANCED",
                        choices=["HIGH_QUALITY", "BALANCED", "RESOURCE_CONSTRAINED"],
                        help="Processing mode balancing fidelity vs CPU/RAM constraints")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Directory to store outputs and reports")
    parser.add_argument("--weights", type=str, default="./models/model_weights", help="Path to U-Net checkpoint")
    parser.add_argument("--dummy", action="store_true", help="Use lightweight mock separator for CI/benchmarking")
    parser.add_argument("--no_plots", action="store_true", help="Skip rendering PNG plots for minimum CPU overhead")

    args = parser.parse_args()

    # Resolve audio path
    audio_path = args.audio
    if not os.path.exists(audio_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, audio_path)
        if os.path.exists(alt):
            audio_path = alt
        else:
            print(f"[-] Error: Audio file '{args.audio}' not found.")
            sys.exit(1)

    print("================================================================================")
    print("   Resource-Aware Embedded Audio Processing & Intelligent Source Separation")
    print("================================================================================")
    print(f"[*] Input File     : {audio_path}")
    print(f"[*] Execution Mode : {args.mode}")
    print(f"[*] Output Dir     : {args.output}")
    print(f"[*] Engine Type    : {'Dummy Mock (Benchmarking)' if args.dummy else '1D U-Net (985K Params)'}")
    print("================================================================================\n")

    # Instantiate Engine
    if args.dummy:
        separator = DummySourceSeparator(latency_simulation=0.05)
    else:
        print("[+] Loading U-Net source separation weights...")
        separator = UNetSourceSeparator(weights_path=args.weights)
        print("[+] Engine initialized and ready.")

    controller = ResourceController(
        separator=separator,
        output_dir=args.output,
        default_mode=ProcessingMode(args.mode)
    )

    source = FileAudioSource(audio_path)

    print("[+] Starting resource-aware audio processing pipeline...")
    saved_paths, report, analysis = controller.process_audio(
        audio_source=source,
        mode=ProcessingMode(args.mode),
        generate_plots=not args.no_plots
    )

    print("\n--------------------------------------------------------------------------------")
    print("                              EXECUTION AUDIT REPORT                            ")
    print("--------------------------------------------------------------------------------")
    print(f"[*] Audio Duration       : {report.audio_duration_seconds:.2f} s ({report.num_segments} segments)")
    print(f"[*] Total Pipeline Time  : {report.total_pipeline_time_seconds:.3f} s")
    print(f"[*] Preprocessing Time   : {report.preprocessing_time_seconds:.3f} s")
    print(f"[*] Neural Inference Time: {report.total_inference_time_seconds:.3f} s")
    print(f"[*] Post-processing Time : {report.postprocessing_time_seconds:.3f} s")
    print(f"[*] Real-Time Factor     : {report.real_time_factor:.3f}x ({'PASS - Real-time capable' if report.is_realtime_capable else 'FAIL - Slower than real-time'})")
    print(f"[*] Peak Memory (RSS)    : {report.peak_memory_mb:.1f} MB (Delta: +{report.memory_delta_mb:.1f} MB)")
    print(f"[*] Memory Profiler      : {report.memory_measurement_method}")
    print("--------------------------------------------------------------------------------")
    print("Saved Artifacts:")
    for key, path in saved_paths.items():
        print(f"  - {key:<20}: {path}")
    print("================================================================================\n")


if __name__ == '__main__':
    main()
