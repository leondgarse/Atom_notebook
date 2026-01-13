```py
import librosa, numpy as np
from tqdm import tqdm

dd = []
music_path = os.path.expanduser('~/Music/Netease/')
for file in tqdm(os.listdir(music_path)):
    y, sr = librosa.load(os.path.join(music_path, file), mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_time = librosa.frames_to_time(beat_frames, sr=sr)
    # beats = beat_time[1:] - beat_time[:-1]
    beats = np.array(beat_time[11:-10] - beat_time[10:-11])
    dd.append({"name": file, "tempo": tempo[0], "mean": 60 / np.mean(beats), "std": np.std(beats), "cv": np.std(beats) * tempo[0] / 60})
df = pd.DataFrame(dd).set_index('name')

np.savez("music_stats.npz", name=df.index.values, **{kk: df[kk].values for kk in df.columns})

data = np.load("music_stats.npz", allow_pickle=True)
aa = {kk: data[kk] for kk in data.keys()}
df = pd.DataFrame(aa).set_index('name')

# Human perception of timing jitter
# Below ~3% (≈10–20 ms at normal tempos): Usually imperceptible, sounds perfectly steady (like quantized drums).
# ~3–6% (≈20–40 ms): Noticeable as a “human feel” or groove, but still stable. Common in live bands.
# >8–10% (≈50–80+ ms): Starts sounding unstable, dragging/rushing becomes obvious. DJs or sequencers would call this “off” or “sloppy.”
df["stability"] = pd.cut(df["cv"] * 100, bins=[0, 3, 6, 100], labels=["tight", "natural groove", "loose"])

print(df[['第一次爱的人' in ii or 'Italia' in ii or 'Contra' in ii or "Mourning" in ii for ii in df.index]].to_markdown())
# | name                                              |   tempo | beat mean |   beat std |   beat cv | beat stability   |
# |:--------------------------------------------------|--------:|--------:|-----------:|----------:|:------------|
# | 王心凌 - 第一次爱的人.mp3                         | 95.7031 | 94.0038 | 0.0137577  | 0.0219442 | tight       |
# | Pnan,HSHK,VodKe,Alpha - Italia e voi（Reset）.mp3 | 95.7031 | 94.9908 | 0.00933088 | 0.0148832 | tight       |
# | Álvaro Soler,David Bisbal - A Contracorriente.mp3 | 95.7031 | 94.9943 | 0.0158946  | 0.0253526 | tight       |
# | Mabanua - Mourning.mp3                            | 95.7031 | 96.986  | 0.0161997  | 0.0258394 | tight       |

print(df[[94 < ii < 96 for ii in df.tempo]].to_markdown())
# | name                                              |   tempo |    mean |        std |        cv | stability      |
# |:--------------------------------------------------|--------:|--------:|-----------:|----------:|:---------------|
# | Aimer - カタオモイ.mp3                            | 95.7031 | 96.6578 | 0.0320813  | 0.0511713 | natural groove |
# | The Wellermen - Wellerman.mp3                     | 95.7031 | 95.9692 | 0.0222013  | 0.0354122 | natural groove |
# | Toni Braxton - Fairy Tale.mp3                     | 95.7031 | 93.977  | 0.0144305  | 0.0230174 | tight          |
# | 孙燕姿 - 180度.mp3                                | 95.7031 | 93.9972 | 0.021781   | 0.0347418 | natural groove |
# | ロザリーナ - えんとつ町のプペル.mp3               | 95.7031 | 95.397  | 0.026044   | 0.0415415 | natural groove |
# | 王心凌 - 第一次爱的人.mp3                         | 95.7031 | 94.0038 | 0.0137577  | 0.0219442 | tight          |
# | 艾索 - 晚安喵.mp3                                 | 95.7031 | 94.9909 | 0.0126423  | 0.0201652 | tight          |
# | 秋山黄色 - 夢の礫.mp3                             | 95.7031 | 95.3651 | 0.0307164  | 0.0489942 | natural groove |
# | Pnan,HSHK,VodKe,Alpha - Italia e voi（Reset）.mp3 | 95.7031 | 94.9908 | 0.00933088 | 0.0148832 | tight          |
# | Ryan - The Beginning.mp3                          | 95.7031 | 95.9886 | 0.00974228 | 0.0155395 | tight          |
# | Fréro Delavega - Price Tag.mp3                    | 95.7031 | 95.9005 | 0.0158368  | 0.0252604 | tight          |
# | IAN POST - Goodbyes.mp3                           | 95.7031 | 96.3606 | 0.0311097  | 0.0496215 | natural groove |
# | ビリー・バンバン - 春夏秋冬.mp3                   | 95.7031 | 94.983  | 0.0184682  | 0.0294577 | tight          |
# | Mabanua - Mourning.mp3                            | 95.7031 | 96.986  | 0.0161997  | 0.0258394 | tight          |
# | majiko - 狂おしいほど僕には美しい.mp3             | 95.7031 | 96.9752 | 0.0330536  | 0.0527222 | natural groove |
# | Mindy Gledhill - All About Your Heart.mp3         | 95.7031 | 95.1377 | 0.0305725  | 0.0487648 | natural groove |
# | 茶太 - Secret.mp3                                 | 95.7031 | 95.0029 | 0.0120705  | 0.0192531 | tight          |
# | 迷宮 - 謎の少女、再び.mp3                         | 95.7031 | 96.1355 | 0.0221969  | 0.0354053 | natural groove |
# | Sebastián Yatra,Reik - Un Año.mp3                 | 95.7031 | 95.641  | 0.023318   | 0.0371935 | natural groove |
# | 李悦君Ericaceae - 梦伴.mp3                        | 95.7031 | 95.3701 | 0.0354522  | 0.0565481 | natural groove |
# | Álvaro Soler,David Bisbal - A Contracorriente.mp3 | 95.7031 | 94.9943 | 0.0158946  | 0.0253526 | tight          |

plt.plot(sorted(df['tempo'].values))
```
