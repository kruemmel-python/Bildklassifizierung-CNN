from moviepy.editor import VideoFileClip, AudioFileClip

def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Merge video and audio files into a single output file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file.
        output_path (str): Path for the output file.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    # Kürze die Audiodatei auf die Länge des Videos
    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)
    
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# Beispiel für die Verwendung der Funktion
if __name__ == "__main__":
    video_path = 'output_wald.mp4'
    audio_path = 'spirit.MP3'
    output_path = 'musikvideo.mp4'

    merge_video_audio(video_path, audio_path, output_path)
