"""Version-locked strip census. No AF model or reference peaks enter eligibility."""
import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ECTOPY_STRICT = {'SVPB', 'VPB'}
ECTOPY_EXTENDED = ECTOPY_STRICT | {'SV Couplet', 'V Couplet', 'TgSV'}


def clinical_group(quality, events):
    if quality == 'N2':
        return 'N2_safety'
    if 'A. Fib.' in events:
        return quality + '_AF'
    if 'A. Flutter' in events:
        return quality + '_AFL_sensitivity'
    if events & ECTOPY_STRICT:
        return quality + '_ectopy_provisional'
    if events & ECTOPY_EXTENDED:
        return quality + '_extended_ectopy_sensitivity'
    if 'AVB' in events:
        return quality + '_AVB_sensitivity'
    return 'excluded_no_primary_reference'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-zip', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.file_digest(args.release_zip.open('rb'), 'sha256').hexdigest()
    if digest != args.expected_sha256:
        raise ValueError('Release ZIP differs from the version-locked bytes')
    for directory in ['manifests', 'cohorts', 'configs/model_contracts', 'reports', 'logs']:
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    rows, identities = [], {}
    with zipfile.ZipFile(args.release_zip) as archive:
        names = archive.namelist()
        base = names[0].split('/')[0] + '/'
        for name in sorted(n for n in names if n.endswith('.hea')):
            header = archive.read(name)
            lines = header.decode().strip().splitlines()
            fields = lines[0].split()
            strip, channels, fs, samples = fields[0], int(fields[1]), float(fields[2]), int(fields[3])
            comment = next(line[1:].strip() for line in lines if line.startswith('# CLASS:'))
            technical, event_text = comment.split('| EVENTS:', 1)
            quality, noise = technical.split('| NOISE:', 1)
            quality = quality.replace('CLASS:', '').strip()
            events = {event.strip() for event in event_text.split('|') if event.strip() not in ('', 'None')}
            stem = str(Path(name).with_suffix(''))
            hashes = {ext: hashlib.sha256(archive.read(f'{stem}.{ext}')).hexdigest() for ext in ['dat', 'atr', 'qrs']}
            lead_lines = lines[1:channels + 1]
            channel_identity = [line.split()[-1] for line in lead_lines]
            # Same bytes are only physically duplicate if calibration, layout
            # and rate also agree. Annotation differences are checked below.
            calibration = [line.split()[1:] for line in lead_lines]
            duplicate_key = hashlib.sha256(json.dumps([hashes['dat'], fs, samples, calibration], sort_keys=True).encode()).hexdigest()
            identities.setdefault(duplicate_key, []).append(strip)
            duration = samples / fs
            group = clinical_group(quality, events)
            rows.append({'strip_id': strip, 'recording_prefix': strip.split('-')[0],
                         'sampling_frequency': fs, 'duration_seconds': duration, 'sample_count': samples,
                         'channel_count': channels, 'channel_identity': json.dumps(channel_identity),
                         'channel_geometry': 'unknown', 'device_type_if_verified': 'unknown',
                         'acquisition_protocol_if_verified': 'unknown', 'patient_id_if_verified': 'unknown',
                         'quality_class': quality, 'noise_type': noise.strip(),
                         'clinical_event_labels': '|'.join(sorted(events)),
                         'af_present': 'A. Fib.' in events, 'afl_present': 'A. Flutter' in events,
                         'ectopy_present': bool(events & ECTOPY_EXTENDED),
                         'strict_svp_pvb_present': bool(events & ECTOPY_STRICT), 'av_block_present': 'AVB' in events,
                         'primary_clinical_cohort': group, 'negative_reference_status': 'provisional_not_explicit_AF_exclusion' if 'ectopy' in group else 'not_applicable',
                         'label_time_scope': 'strip_context_not_window_adjudicated',
                         'duplicate_group_id': duplicate_key, 'duplicate_canonical': True,
                         'duplicate_annotation_conflict': False,
                         'eligible_10s': duration >= 10 and group != 'excluded_no_primary_reference',
                         'eligible_30s': duration >= 30 and group != 'excluded_no_primary_reference',
                         'duration_support_10s': duration >= 10, 'duration_support_30s': duration >= 30,
                         'exclusion_reason': 'no_explicit_primary_reference' if group == 'excluded_no_primary_reference' else '',
                         'header_sha256': hashlib.sha256(header).hexdigest(),
                         'signal_sha256': hashes['dat'], 'atr_sha256': hashes['atr'], 'qrs_sha256': hashes['qrs'],
                         'raw_header_text': header.decode(), 'header_archive_member': name,
                         'atr_archive_member': stem + '.atr', 'qrs_archive_member': stem + '.qrs',
                         'original_global_comment': comment})
        listed_ids = {Path(line).name for line in archive.read(base + 'RECORDS').decode().splitlines() if line.strip()}
        checksum_failures = []
        checksum_lines = archive.read(base + 'SHA256SUMS.txt').decode().splitlines()
        for line in checksum_lines:
            expected, member = line.split(maxsplit=1)
            member = member.lstrip('*')
            actual = hashlib.sha256(archive.read(base + member)).hexdigest()
            if expected != actual:
                checksum_failures.append(member)
    if checksum_failures:
        raise ValueError(f'Official checksum mismatch: {checksum_failures}')
    frame = pd.DataFrame(rows)
    for key, ids in identities.items():
        selected = frame.duplicate_group_id == key
        if len(ids) < 2:
            continue
        labels = frame.loc[selected, ['quality_class', 'clinical_event_labels']].drop_duplicates()
        conflict = len(labels) > 1
        canonical = min(ids)
        frame.loc[selected, 'duplicate_annotation_conflict'] = conflict
        frame.loc[selected, 'duplicate_canonical'] = frame.loc[selected, 'strip_id'].eq(canonical)
        excluded = selected if conflict else selected & ~frame.strip_id.eq(canonical)
        frame.loc[excluded, ['eligible_10s', 'eligible_30s']] = False
        frame.loc[excluded, 'exclusion_reason'] = 'duplicate_annotation_conflict' if conflict else 'duplicate_noncanonical'
    frame.to_csv(args.output / 'manifests/segment_manifest.csv', index=False)
    frame.to_parquet(args.output / 'manifests/segment_manifest.parquet', index=False)
    # Exclusive groups keep AF+ectopy out of the provisional negative cohort.
    clinical = frame[['strip_id', 'recording_prefix', 'primary_clinical_cohort', 'af_present', 'afl_present',
                      'ectopy_present', 'negative_reference_status', 'eligible_10s', 'eligible_30s', 'exclusion_reason']]
    clinical.to_csv(args.output / 'cohorts/clinical_cohorts.csv', index=False)
    flow = []
    for minimum in [0, 10, 30]:
        for (quality, group), subset in frame.groupby(['quality_class', 'primary_clinical_cohort']):
            duration = subset[subset.duration_seconds >= minimum]
            kept = duration[duration.exclusion_reason == '']
            flow.append({'minimum_seconds': minimum, 'quality_class': quality, 'cohort': group,
                         'before_dedup_strips': len(duration), 'before_dedup_recording_prefixes': duration.recording_prefix.nunique(),
                         'retained_strips': len(kept), 'retained_recording_prefixes_NOT_PATIENTS': kept.recording_prefix.nunique(),
                         'retained_one_channel': int((kept.channel_count == 1).sum()),
                         'retained_two_channels': int((kept.channel_count == 2).sum()),
                         'duplicate_exclusions': int(duration.exclusion_reason.str.startswith('duplicate').sum())})
    pd.DataFrame(flow).to_csv(args.output / 'cohorts/exclusion_flow.csv', index=False)
    integrity = {'release': 'Noise Holter v1.0.0', 'archive': str(args.release_zip), 'archive_sha256': digest,
                 'physical_strips': len(frame), 'listed_strips': len(listed_ids),
                 'missing_listed_strips': sorted(listed_ids - set(frame.strip_id)),
                 'recording_prefixes_NOT_PATIENTS': frame.recording_prefix.nunique(),
                 'verified_patient_ids': 0, 'official_checksum_entries_verified': len(checksum_lines),
                 'checksum_failures': checksum_failures,
                 'physical_duplicate_groups': sum(len(ids) > 1 for ids in identities.values()),
                 'duplicate_noncanonical_strips': int((frame.exclusion_reason == 'duplicate_noncanonical').sum()),
                 'duplicate_conflict_strips': int(frame.duplicate_annotation_conflict.sum()),
                 'eligibility_definition': 'duration plus explicit cohort and duplicate policy; model-specific observed-RR support is computed separately without reference annotations',
                 'patient_protocol_device_mapping': 'unknown; no inference from prefix, duration or channel A/B',
                 'manifest_sha256': hashlib.file_digest((args.output / 'manifests/segment_manifest.parquet').open('rb'), 'sha256').hexdigest()}
    (args.output / 'manifests/release_integrity.json').write_text(json.dumps(integrity, indent=2, default=int))
    print(json.dumps({key: value for key, value in integrity.items() if key != 'missing_listed_strips'}, indent=2, default=int))


if __name__ == '__main__':
    main()
