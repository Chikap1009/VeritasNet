# src/narrative_bias/video_transcriber.py

import os
import yt_dlp  # For downloading videos
import whisper  # For transcription
import re  # For extracting video ID


def extract_video_id(youtube_url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, youtube_url)
    if match:
        return match.group(1)
    return None


def download_youtube_video_with_ytdlp(youtube_url, output_path='.', filename=None):
    os.makedirs(output_path, exist_ok=True)

    try:
        ydl_opts_info = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'unknown_video')
            video_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '.', '_')).rstrip()
    except Exception as e:
        print(f"Warning: Could not fetch video title: {e}")
        video_title = "unknown_video"

    final_filename = filename if filename else video_title
    final_filename = "".join(c for c in final_filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

    file_extension = 'mp4'
    download_target_path = os.path.join(output_path, f"{final_filename}.{file_extension}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': download_target_path,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',
    }

    print(f"Downloading '{final_filename}.{file_extension}' from YouTube using yt-dlp...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"Downloaded: {download_target_path}")
        return download_target_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


def transcribe_audio_with_whisper(audio_path, model_size="base"):
    try:
        print(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        print(f"Transcribing '{os.path.basename(audio_path)}'...")
        result = model.transcribe(audio_path)
        print("Transcription complete.")
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def process_video_for_transcript(youtube_url, video_output_dir="./downloaded_videos", transcript_output_dir="./transcripts"):
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(transcript_output_dir, exist_ok=True)

    video_id = extract_video_id(youtube_url)
    if not video_id:
        print(f"Could not extract video ID from URL: {youtube_url}")
        return None, None, None

    video_title = f"youtube_video_{video_id}"
    try:
        ydl_opts_info = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', video_title)
            video_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '.', '_')).rstrip()
    except Exception as e:
        print(f"Warning: Could not fetch video title: {e}")

    downloaded_video_path = download_youtube_video_with_ytdlp(youtube_url, output_path=video_output_dir, filename=video_title)

    transcript_text = None
    transcript_file_path = None

    if downloaded_video_path:
        transcript_text = transcribe_audio_with_whisper(downloaded_video_path, model_size="base")
        if transcript_text:
            transcript_filename = f"{video_title}_transcript_whisper.txt"
            transcript_filename = "".join(c for c in transcript_filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            transcript_file_path = os.path.join(transcript_output_dir, transcript_filename)
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"Transcript saved to: {transcript_file_path}")
        else:
            print("❌ Failed to generate transcript.")
    else:
        print("❌ Failed to download video. Cannot transcribe.")

    return downloaded_video_path, transcript_file_path, transcript_text


# ───────────────────────────────────────────────
# ✅ CLI + ANALYSIS ENTRYPOINT
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VeritasNet Video Transcriber + Analyzer")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL to process")
    parser.add_argument("--analyze", action="store_true", help="Run bias/sentiment/framing analysis after transcription")

    args = parser.parse_args()

    print("\n📥 Step 1: Downloading and Transcribing...")
    video_path, transcript_path, transcript_text = process_video_for_transcript(args.url)

    if transcript_text:
        print("\n📄 Transcript Preview (first 500 chars):")
        print(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)

        if args.analyze:
            print("\n🔍 Step 2: Analyzing Transcript via VeritasNet Pipeline...")
            try:
                from predict import analyze_transcript
                results = analyze_transcript(transcript_path)
                if results:
                    print("\n✅ Analysis Complete.")
                else:
                    print("\n❌ Analysis Failed.")
            except Exception as e:
                print(f"Error running prediction pipeline: {e}")
    else:
        print("❌ Transcript generation failed.")