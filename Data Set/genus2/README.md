# Genus-two production dataset

The large production dataset is stored on Google Drive rather than in this
Git repository. The Google Drive folder contains a gzip-compressed tar archive
split into two parts to stay below Drive's per-upload limit:

- [Google Drive folder](https://drive.google.com/drive/folders/1A0BLqSHjMQA8Z34ttWJDwcJd17BSVow5)
- [Part 00 (90 MiB)](https://drive.google.com/file/d/1Ftb4T1r5_MyRrzyhxSKLUvAHoY8sW5lS/view?usp=sharing)
- [Part 01 (36,044,531 bytes)](https://drive.google.com/file/d/1swH6GFUy8ctnLJz_zqoHaxZHDbghUObT/view?usp=sharing)

## Integrity checks

| File | SHA-256 |
| --- | --- |
| `Type0B-Matrix-genus2-dataset-2026-08-01.tar.gz.part-00` | `2b1ed24eeecc3281262dda1cc4cd06a9a59bae30e0930325bc04810a47622ac7` |
| `Type0B-Matrix-genus2-dataset-2026-08-01.tar.gz.part-01` | `9656f65621a927dbd267cf1ec3c7500d6609c17209a7f7b2de665a5e495fa899` |
| Reassembled `.tar.gz` archive | `c999485028cab2c8d9906a9847c363284f2afc76560c1ce1e033c2ac5d9434c7` |

On macOS or Linux, download both parts into one directory and run:

```bash
cat Type0B-Matrix-genus2-dataset-2026-08-01.tar.gz.part-* \
  > Type0B-Matrix-genus2-dataset-2026-08-01.tar.gz
shasum -a 256 Type0B-Matrix-genus2-dataset-2026-08-01.tar.gz
tar -xzf Type0B-Matrix-genus2-dataset-2026-08-01.tar.gz -C "Data Set"
```

Extraction creates `Data Set/genus2/`, which is the target of the relative
compatibility symlink at `Code/python/data`.
