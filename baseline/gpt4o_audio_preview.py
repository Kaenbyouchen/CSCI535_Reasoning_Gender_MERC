import sys

from gpt52 import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "yaml/gpt4o_audio_preview_MERC.yaml"])
    main()
