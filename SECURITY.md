# Security and Responsible Disclosure

FHEViT is an experimental research artifact, not a clinical or production security
boundary. It does not establish that decrypted representations protect image
privacy, and it must not be used for diagnosis or patient care.

Please do not report patient data, access tokens, private model files, or other
sensitive material in a public issue. Remove local-only assets before collecting a
reproduction. For a suspected release or dependency problem, contact the
repository maintainer privately through the GitHub profile for
[`FinalSunFlower`](https://github.com/FinalSunFlower) and include a minimal,
non-sensitive reproduction.

The source release deliberately excludes datasets, patient-level outputs, model
snapshots, checkpoints, feature caches, private ledgers, and local environments.
See `.gitignore`, `data/README.md`, `models/README.md`, and
`docs/ENVIRONMENT.md`.
