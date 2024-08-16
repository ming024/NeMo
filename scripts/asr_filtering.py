import json
import re
import argparse
import os
from dataclasses import dataclass
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.random.manual_seed(0)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
dictionary = {c: i for i, c in enumerate(labels)}


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        if t <= 0:
            return None

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_manifest',
        type=str,
        default="/workspace/data/s2s/es/manifest_es_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec.json",
    )
    parser.add_argument(
        '--output_manifest',
        type=str,
        default="/workspace/data/s2s/es/manifest_es_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec_forced_ali_score.json",
    )
    parser.add_argument(
        '--audio_folder',
        type=str,
        default="/workspace/data/s2s/es/target_wav/",
    )
    args = parser.parse_args()

    metadata = []
    scores = []
    with open(args.input_manifest, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            transcript = line["question"].strip("Phoneme TTS ")

            audio_path = os.path.join(args.audio_folder, "Sample_Audios_0", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_1", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_2", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_3", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_4", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_5", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_6", line["target_wav"].split('/')[-1])
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_folder, "Sample_Audios_7", line["target_wav"].split('/')[-1])
            
            try:
                SPEECH_FILE = torchaudio.utils.download_asset(audio_path)
            except:
                ali_score = 0.
                scores.append(ali_score)
                line["ali_score"] = ali_score
                metadata.append(line)
                continue
            
# transcript = "O Hara is believed to have received no formal training and was self-taught!"
            transcript = "|" + re.sub('[^a-zA-Z]+', '|', transcript) + "|"
            transcript = re.sub('[|]+', '|', transcript).upper()

            with torch.inference_mode():
                waveform, _ = torchaudio.load(SPEECH_FILE)
                emissions, _ = model(waveform.to(device))
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            tokens = [dictionary[c] for c in transcript]

            trellis = get_trellis(emission, tokens)

            path = backtrack(trellis, emission, tokens)
            if path is not None:
                segments = merge_repeats(path)
                seg_scores = [seg.score for seg in segments]
                ali_score = np.mean(seg_scores)
            else:
                # Forced alignment fails
                ali_score = 0.
            scores.append(ali_score)

            line["ali_score"] = ali_score
            metadata.append(line)
            
    with open(args.output_manifest, "w") as f:
        for line in metadata:
            f.write(json.dumps(line)+"\n")

if __name__ == '__main__':
    main()