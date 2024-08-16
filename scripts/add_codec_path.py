import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_manifest',
        type=str,
        default="/workspace/data/s2s/es/manifest_es_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s.json",
    )
    parser.add_argument(
        '--output_manifest',
        type=str,
        default="/workspace/data/s2s/es/manifest_es_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec.json",
    )
    parser.add_argument(
        '--codec_base_dir',
        type=str,
        default="/workspace/data/s2s_codec/es/codecs/nemo_codec_bw_6.0/",
    )
    args = parser.parse_args()

    lines = []
    with open(args.input_manifest, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    
    n_failures = 0
    with open(args.output_manifest, 'w') as f:
        for line in lines:
            target_wav = line["target_wav"]
            codec_path = "_".join(target_wav.split("/")[-2:]).replace("Sample_", "Sample_Audios_").replace(".wav", ".pt")
            codec_path =  os.path.join(args.codec_base_dir, codec_path)
            line["target_codec"] = codec_path
            if not os.path.exists(codec_path):
                n_failures += 1
                continue

            f.write(json.dumps(line)+'\n')
    print(n_failures)

if __name__ == '__main__':
    main()