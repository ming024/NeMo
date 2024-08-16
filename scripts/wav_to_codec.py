import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
# from encodec import EncodecModel
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.transformer import mask_from_lens
from nemo.core.classes import Dataset
from nemo.utils import logging

try:
    from models.soundstream import SoundStream
except:
    logging.warning("SoundStream not found, uniaudio cannot be used")

try:
    import dac
except:
    logging.warning("DAC not found")


class AudioDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        base_data_dir,
        min_duration=1.0,
        max_duration=22.0,
        sample_rate=24000,
        min_snr_db=0,
        max_snr_db=5,
        pad_multiple=320,
        audio_type="actual", # actual or noise or silence
    ):
        self.data = []
        for manifest_path in manifest_paths:
            with open(manifest_path, "r") as f:
                for line in tqdm(f):
                    record = json.loads(line)
                    if 'answer_duration' not in record:
                        record['answer_duration'] = record['duration']

                    if record['duration'] < min_duration or record['duration'] > max_duration:
                        continue

                    # Changing target wav paths to local paths
                    record["target_wav"] = os.path.join(base_data_dir, '/'.join(record["target_wav"].replace("Sample_", "Sample_Audios_").split('/')[-2:]))
                    
                    if self._is_record_valid(record):
                        self.data.append(record)

        self.sample_rate = sample_rate
        self.audio_type = audio_type

        self.pad_multiple = pad_multiple

        self.base_data_dir = base_data_dir

    def _is_record_valid(self, record):
        try:
            sf.read(record["target_wav"])
            # sf.read(record["context"])
            return True
        except:
            print("Skipping invalid record", record["target_wav"])
            return False
        
    def __len__(self):
        return len(self.data)

    def _get_wav_from_filepath(self, audio_filepath):
        if self.audio_type == "noise" or self.audio_type == "silence":
            # Create a 6 second noise audio
            if self.audio_type == "noise":
                audio_samples = np.random.normal(0, 1, 6 * self.sample_rate)
            else:
                audio_samples = np.zeros(6 * self.sample_rate)
            audio = torch.tensor(audio_samples).float()
            audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
            audio_length = torch.tensor(audio.size(0)).long()

            return audio, audio_length
        elif self.audio_type == "actual":
            features = AudioSegment.segment_from_file(
                audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=False,
            )
            audio_samples = features.samples
            audio = torch.tensor(audio_samples)
            audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
            audio_length = torch.tensor(audio.size(0)).long()

            return audio, audio_length
        
        else:
            raise ValueError("Unknown audio type {}".format(self.audio_type))

    def pad_collate_fn(self, batch):
        final_batch = {}
        for row in batch:
            for key in row:
                if key not in final_batch:
                    final_batch[key] = []
                final_batch[key].append(row[key])

        max_audio_len = max([_audio_len.item() for _audio_len in final_batch["audio_len"]])

        audios_padded = []
        for audio in final_batch["audio"]:
            audio_padded = torch.nn.functional.pad(audio, (0, max_audio_len - audio.size(0)), value=0)
            audios_padded.append(audio_padded)

        final_batch["audio"] = audios_padded

        non_tensor_keys = [
            "audio_filepath",
            "text",
            "rel_audio_path_as_text_id",
        ]

        for key in final_batch:
            if key not in non_tensor_keys:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["target_wav"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

        # Avoid fixed seed
        random.seed(time.time())

        audio, audio_length = self._get_wav_from_filepath(sample["target_wav"])

        return {
            "audio": audio,
            "audio_len": audio_length,
            "rel_audio_path_as_text_id": rel_audio_path_as_text_id,
            "audio_filepath": sample["target_wav"],
            "text": sample["text"],
        }


def save_batch_audios(batch, bidx, temp_dir, codec_model, codec_model_type='encodec', codec_model_sample_rate=24000):
    for sidx in range(batch["audio"].shape[0]):
        sample_audio = batch["audio"][sidx]
        sample_audio_len = batch["audio_len"][sidx].item()
        sample_audio = sample_audio[:sample_audio_len]

        # Save sample_audio
        sample_audio_path = os.path.join(temp_dir, f"{bidx}_{sidx}_sample.wav")
        torchaudio.save(sample_audio_path, sample_audio[None].cpu(), codec_model_sample_rate)

def main():
    parser = argparse.ArgumentParser(description='Create multiple tasks')
    parser.add_argument(
        '--manifest_paths',
        type=str,
        default="/workspace/data/s2s/es/manifest_es_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s.json",
    )
    parser.add_argument(
        '--base_data_dir',
        type=str,
        default="/workspace/data/s2s/es/target_wav/",
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default='/workspace/data/s2s_codec/es/')
    parser.add_argument('--codec_model_path', type=str, default='/workspace/code/SpeechCodec_2402.nemo')
    parser.add_argument('--codec_bw', type=float, default=6.0)  # 6 for 8 codebooks, 1.5 for 3 codebooks
    parser.add_argument('--codec_model', type=str, default='nemo_codec')  # encodec, uniaudio_codec, dac or nemo_codec
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split_into_train_val', action='store_true')
    parser.add_argument('--num_val_records', type=int, default=500)
    parser.add_argument('--audio_type', type=str, default='actual')  # actual, noise or silence
    args = parser.parse_args()

    if args.codec_model == 'encodec':
        codec_model = EncodecModel.encodec_model_24khz()
        codec_model.set_target_bandwidth(6.0)
        codec_model.cuda()
        codec_model.eval()
        codec_model_sample_rate = 24000
        codec_model_downsampling_factor = 320.0
    elif args.codec_model == 'uniaudio_codec':
        codec_config_path = os.path.join(os.path.dirname(args.codec_model_path), 'config.yaml')
        codec_config = OmegaConf.load(codec_config_path)
        codec_model = eval(codec_config.generator.name)(**codec_config.generator.config)
        codec_parameter_dict = torch.load(args.codec_model_path)
        codec_model.load_state_dict(codec_parameter_dict['codec_model'])  # load model
        codec_model = codec_model.cuda()
        # codec_model.eval()
        codec_model_sample_rate = 16000
        codec_model_downsampling_factor = 320.0
    elif args.codec_model == 'dac':
        model_path = args.codec_model_path
        codec_model = dac.DAC.load(model_path)
        codec_model.to('cuda')
        codec_model_sample_rate = 44100
        codec_model_downsampling_factor = 512.0
    elif args.codec_model == 'nemo_codec':
        model_path = args.codec_model_path
        codec_model = AudioCodecModel.restore_from(model_path)
        codec_model.to('cuda')
        codec_model.eval()
        codec_model_sample_rate = 22050
        codec_model_downsampling_factor = 256.0
    else:
        raise ValueError("Unknown codec model {}".format(args.codec_model))

    dataset = AudioDataset(
        manifest_paths=[args.manifest_paths],
        base_data_dir=args.base_data_dir,
        sample_rate=codec_model_sample_rate,
        pad_multiple=int(codec_model_downsampling_factor),
        audio_type=args.audio_type,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.pad_collate_fn, shuffle=False, num_workers=8,
    )

    _exp_name = "{}_bw_{}".format(args.codec_model, args.codec_bw)
    temp_dir = os.path.join(args.out_dir, "temp_{}".format(_exp_name))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    codec_base_dir = os.path.join(args.out_dir, "codecs")
    manifest_dir = os.path.join(args.out_dir, "manifests")

    audiocodec_out_dir = os.path.join(codec_base_dir, _exp_name)

    if not os.path.exists(audiocodec_out_dir):
        os.makedirs(audiocodec_out_dir)

    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)

    all_tasks_records = []

    for bidx, batch in enumerate(tqdm(dataloader)):
        # print("bidx", bidx+1, "of", len(dataloader))

        audio_len_mask = mask_from_lens(batch["audio_len"])

        cuda_keys = ['audio', 'audio_len']
        for key in cuda_keys:
            batch[key] = batch[key].cuda()
        with torch.no_grad():
            if args.codec_model == 'encodec':
                original_codec_codes = codec_model.encode(batch["audio"].unsqueeze(1))[0][0]
            elif args.codec_model == 'uniaudio_codec':
                original_codec_codes = codec_model.encode(
                    batch["audio"].unsqueeze(1) * codec_config.audio_norm_scale, target_bw=args.codec_bw
                ).permute(1, 0, 2)
                print("original_codec_codes", original_codec_codes.shape)
            elif args.codec_model == 'dac':
                # z, codes, latents, _, _ = model.encode(x)
                _, original_codec_codes, _, _, _ = codec_model.encode(batch["audio"].unsqueeze(1))
            elif args.codec_model == 'nemo_codec':
                original_codec_codes, _ = codec_model.encode(audio=batch["audio"], audio_len=batch["audio_len"])
            else:
                raise ValueError("Unknown codec model {}".format(args.codec_model))

        # codec_codes = transformer_encodec_model.encode(batch["audio"].unsqueeze(1), audio_len_mask, bandwidth=6.0)
        target_codecs = []
        for sidx in range(batch['audio'].shape[0]):

            codec_len = math.ceil(batch['audio_len'][sidx].item() / codec_model_downsampling_factor)
            sample_codec_codes = original_codec_codes[sidx][:, :codec_len]
            target_codecs.append(sample_codec_codes)

            example_name = batch['rel_audio_path_as_text_id'][sidx]

            target_codec_filepath = os.path.join(audiocodec_out_dir, "{}.pt".format(example_name))
            torch.save(sample_codec_codes.cpu().type(torch.int16), target_codec_filepath)

        if bidx == 0:
            save_batch_audios(batch, bidx, temp_dir, codec_model, args.codec_model, codec_model_sample_rate)

if __name__ == '__main__':
    main()