# Stage2 first-submission component v1

This bundle implements the official `predict_stage2(data_dir, model_dir)` interface.
It is intended to be merged with the separately developed Stage1 and Stage3 entry
points before creating the competition `submit.zip`.

## Outputs

- `collision_frame`: camera-motion-compensated impact score, with physical-contact
  offset correction.
- `entry_frame`: first near-side wheel contact with a temporally tracked ego-lane
  boundary; perspective-corridor fallback is used when lane markings are absent.
- `entry_side`: inferred primarily from lateral travel direction.
- `evasion_space`: free drivable width beside the victim at impact.

All frame outputs are original filename frame numbers. The implementation always
returns integer `evasion_space`, `LEFT/RIGHT`, and an `entry_frame` no later than the
collision frame.

## Offline model

YOLOP `End-to-end.pth` and the minimal YOLOP runtime are included under
`model/stage2`. No inference-time download or external API is used. YOLOP is MIT
licensed; the included `YOLOP_LICENSE` must remain with redistributed code.

## Known limitations

- Public Stage2 examples label collision only. Entry, direction, and evasion rules
  are definition-driven rather than supervised by public ground truth.
- If lane segmentation fails, the perspective corridor assumes a roughly centered
  dashcam.
- The collision signal was validated on four public direct-ego crashes; the public
  third-party-only crash is outside the stated hidden-set condition and fails.
