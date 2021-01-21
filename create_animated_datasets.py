import datetime as dt
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from IPython.display import Video
from pathlib import Path

from matplotlib import animation
from matplotlib import patches
# from tqdm.notebook import tqdm

warnings.simplefilter("ignore")


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='forestgreen', zorder=0)  # changed the field color to forestgreen

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax



track_data = pd.read_csv(r"E:\kaggle\nfl-impact-detection\train_player_tracking.csv")
track_data["time"] = pd.to_datetime(track_data["time"])
track_data["color"] = track_data["player"].map(lambda x: "black" if "H" in x else "white")
track_data.head()


train_labels = pd.read_csv(r"E:\kaggle\nfl-impact-detection\train_labels.csv")
train_labels.head()


# Create train_player_tracking.csv with impact annotation
def make_alignment(train_track: pd.DataFrame, train_label: pd.DataFrame, video_dir: Path, game_key: int, play_id: int):
    play_track = train_track.query(f"gameKey == {game_key} & playID == {play_id}")
    play_label = train_label.query(f"gameKey == {game_key} & playID == {play_id}")

    play_track["impact"] = 0
    play_track["impactType"] = ""
    play_track["confidence"] = 0
    play_track["visibility"] = 0

    snap_frame = play_track.query("event == 'ball_snap'")
    snap_time = snap_frame["time"].iloc[0]

    video_name = f"{game_key}_{str(play_id).rjust(6, '0')}_Endzone.mp4"
    video = cv2.VideoCapture(str(video_dir / video_name))

    fps = video.get(cv2.CAP_PROP_FPS)
    nframes = play_label.frame.nunique()

    snap_time -= dt.timedelta(seconds=1.0 / fps * 10)

    duration = nframes / fps
    end_time = snap_time + dt.timedelta(seconds=duration)

    play = play_track.loc[(play_track["time"] >= snap_time) & (play_track["time"] < end_time)].copy()

    impact_frames = play_label.query("impact == 1 & view == 'Endzone'")
    for _, row in impact_frames.iterrows():
        frame = row.frame
        label = row.label
        time_from_start = frame / fps
        time = snap_time + dt.timedelta(seconds=time_from_start)

        abs_timedelta = abs(play["time"] - time).dt.total_seconds()
        min_abs_timedelta = abs_timedelta.min()
        impact_point_index = play[abs_timedelta == min_abs_timedelta].query(
            f"player == '{label}'").index[0]
        play.loc[impact_point_index, "impact"] = 1
        play.loc[impact_point_index, "impactType"] = row.impactType
        play.loc[impact_point_index, "confidence"] = row.confidence
        play.loc[impact_point_index, "visibility"] = row.visibility
    play = play.reset_index(drop=False)
    return play


pairs = track_data.groupby(["gameKey", "playID"]).count().index.tolist()
video_dir = Path(r"E:\kaggle\nfl-impact-detection\train")

play_trackings = []
for game_key, play_id in pairs:
    play_trackings.append(make_alignment(track_data, train_labels, video_dir, game_key, play_id))

annotated_trackings = pd.concat(play_trackings, axis=0).reset_index(drop=True)
annotated_trackings.head(10)

annotated_trackings.query("impact == 1 & gameKey == 57583 & playID == 82")
train_labels.query("impact == 1 & gameKey == 57583 & playID == 82 & view == 'Endzone'")
len(annotated_trackings.query(
    "impact == 1 & gameKey == 57583 & playID == 82")), len(train_labels.query(
    "impact == 1 & gameKey == 57583 & playID == 82 & view == 'Endzone'"))

annotated_trackings.to_csv(r"E:\kaggle\nfl-impact-detection\train_player_tracking_annotated.csv", index=False)
annotated_trackings.impactType.unique()

