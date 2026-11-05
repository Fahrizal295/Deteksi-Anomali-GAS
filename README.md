# Purwarupa Deteksi Anomali Gas CO dan CO2

Repository ini memisahkan akuisisi, pemrosesan, model, dan dashboard. Sensor aktif adalah Winsen ZE07-CO melalui UART dan DFRobot Gravity STCC4 SEN0678 melalui I2C. SCD41 tidak digunakan oleh alur aktif.

## Arsitektur

- `src/sensors/`: adapter ZE07-CO dan STCC4. Adapter tidak membuka hardware ketika modul diimpor.
- `ZE07COBackgroundReader`: thread khusus active-upload UART yang menyimpan frame valid terbaru; scheduler 1 Hz hanya mengambil snapshot non-blocking.
- `src/acquisition_service.py`: proses terpisah yang membaca DUMMY atau SENSOR setiap 1 detik, memvalidasi status, menjalankan SMA causal 5 sampel, inference Isolation Forest, dan menulis CSV. Proses ini tidak melatih model secara otomatis.
- `src/acquisition_service.py --mode collect-baseline`: proses akuisisi baseline sensor-only; tidak memuat, melatih, atau menjalankan Isolation Forest.
- `src/sensor_pipeline.py`: pipeline validasi -> SMA -> fitur `co_ma`, `co2_ma` -> inference -> warning.
- `src/offline_training.py`: review dan training offline dari CSV baseline normal yang dikumpulkan terpisah.
- `src/dashboard.py`: Streamlit read-only terhadap CSV; browser refresh tidak menginisialisasi sensor atau melatih model.
- `src/sensors/vendor/DFRobot_STCC4.py`: kode driver DFRobot resmi yang divendorkan dengan atribusi dan lisensi MIT.

Threshold 35 ppm CO dan 1000 ppm CO2 adalah parameter simulasi operasional, bukan klaim batas kesehatan sesaat. Threshold alert memakai nilai sensor valid sebelum SMA. Data invalid, termasuk status STCC4 256/0x0100 yang belum terverifikasi pada unit fisik, tidak masuk SMA atau model.

Dashboard menampilkan dua indikator berbeda: `if_status` (`NOT READY`, `NORMAL`, `ANOMALY`) adalah hasil Isolation Forest, sedangkan `safety_status` (`NORMAL`, `EARLY WARNING`, `THRESHOLD ALERT`, `SENSOR ERROR`) mengikuti kebijakan operasional. Anomali statistik tidak otomatis berarti kondisi berbahaya.

## Mode simulasi Windows

Install dependensi:

```powershell
python -m pip install -r requirements.txt
```

Jika sudah memiliki baseline normal sensor, tinjau lalu latih model secara offline:

```powershell
python -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --review
python -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --approve
```

Jalankan acquisition di terminal pertama:

```powershell
python -m src.acquisition_service --mode dummy --scenario normal
```

Jalankan dashboard di terminal kedua:

```powershell
python -m streamlit run src/dashboard.py --server.port 8501
```

Buka `http://localhost:8501`. Skenario dapat dipilih melalui kontrol dashboard; tombol Start/Pause/Reset mengirim perintah ke acquisition service melalui file kontrol. Acquisition service menolak berjalan jika model belum dilatih offline.

Smoke test singkat dengan model yang sudah tersedia:

```powershell
python -m src.acquisition_service --mode dummy --scenario sudden_spike --max-samples 8 --output data/smoke.csv --model data/models/isolation_forest.joblib
```

## Deployment Raspberry Pi 4

Aktifkan I2C, pastikan ZE07-CO terhubung ke UART `/dev/serial0` pada 9600 baud, dan verifikasi STCC4 pada bus 1/alamat `0x64` (`i2cdetect -y 1`). Install dependensi, review baseline, lalu jalankan acquisition sebagai proses terpisah:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --review
python3 -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --approve
python3 -m src.acquisition_service --mode sensor
```

Untuk akses dashboard dari laptop/HP pada jaringan yang sama:

```bash
python3 -m streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Kemudian buka `http://ALAMAT_IP_RASPBERRY_PI:8501`. Jangan menjalankan `--mode sensor` di Windows.

## Mengumpulkan baseline langsung dari sensor

Mode baseline tidak membutuhkan model yang sudah dilatih. Jalankan hanya pada Raspberry Pi setelah ZE07-CO dan STCC4 terhubung:

```bash
python3 -m src.acquisition_service \
  --mode collect-baseline \
  --duration 1800 \
  --output data/baseline_sensor_normal.csv
```

Sampling menggunakan jadwal monotonic nominal 1 detik. File dibuka dalam mode append dan header/schema yang tidak cocok akan ditolak untuk mencegah pencampuran data. Data yang tersimpan mencakup timestamp siklus, timestamp sensor individual (`co_timestamp`, `co2_timestamp`), waktu UTC dan monotonic awal/akhir siklus, CO, CO2, temperatur, kelembapan, status STCC4, validitas dan error masing-masing sensor (`co_valid`, `co_error`, `co2_valid`, `co2_error`), validitas gabungan, source, serta diagnostik JSON (`co_diagnostics`, `co2_diagnostics`). Retry ZE07-CO maksimal dua percobaan dengan timeout serial 250 ms; STCC4 maksimal satu retry setelah 150 ms sesuai datasheet. Tidak ada nilai sintetis atau pengisian NaN. `Ctrl+C` menghentikan loop dengan aman dan mempertahankan baris yang sudah tertulis.
Snapshot ZE07-CO menggunakan `ze07_freshness_timeout_seconds=1.5` sebagai freshness timeout awal. Kolom `co_frame_id`, `co_age_seconds`, `co_is_fresh`, `co_is_expired`, dan `co_frame_state` membedakan frame baru, frame yang dipakai ulang dalam batas freshness, dan frame kedaluwarsa. Frame `REUSED` tidak boleh dihitung sebagai observasi CO independen saat menyiapkan baseline training.

Schema baseline dengan kolom diagnostik adalah schema baru. File baseline lama tidak ditimpa atau dicampur; gunakan nama file pengujian baru jika header lama ditolak. Modul training offline tetap hanya membutuhkan kolom inti `timestamp`, `co_ppm`, dan `co2_ppm`, sehingga file baseline baru tetap dapat ditinjau sebelum training.

## Pengujian timing 180 detik

Setelah warm-up sensor, jalankan pada Raspberry Pi dengan file baru:

```bash
python3 -m src.acquisition_service \
  --mode collect-baseline \
  --duration 180 \
  --output data/baseline_sensor_180s_v2.csv
```

Evaluasi jumlah frame unik dan frame reuse menggunakan kolom `co_frame_id`, `co_is_fresh`, `co_is_expired`, dan `co_age_seconds`. Bandingkan juga `cycle_start_monotonic`, `cycle_end_monotonic`, serta `co_timestamp`. Pengujian ini tidak melatih atau menjalankan Isolation Forest.

Respons STCC4 juga divalidasi dengan CRC-8 per word sesuai protokol (polynomial `0x31`, initial `0xFF`). Respons CRC invalid diperlakukan sebagai pembacaan tidak valid dan dilaporkan sebagai error STCC4.

## Diagnostik raw STCC4 pada Raspberry Pi

Skrip [tests/hardware/diagnose_stcc4.py](tests/hardware/diagnose_stcc4.py) terpisah dari acquisition service dan tidak mengubah aturan validasi produksi. Skrip membaca 60 respons STCC4 pada interval nominal 1 detik, menyimpan raw response 12 byte, status desimal/hexadecimal, status MSB/LSB, CRC setiap blok, serta nilai CO2, temperatur, dan kelembapan ke CSV baru.

Jalankan setelah I2C bus 1 dan sensor pada alamat `0x64` diverifikasi:

```bash
python3 tests/hardware/diagnose_stcc4.py \
  --samples 60 \
  --output data/stcc4_diagnostic_60.csv
```

File output tidak ditimpa. Gunakan nama file baru untuk setiap sesi. Jika ingin menjalankan self-test terdokumentasi sebelum pengukuran, gunakan opsi eksplisit berikut:

```bash
python3 tests/hardware/diagnose_stcc4.py \
  --samples 60 \
  --self-test \
  --output data/stcc4_diagnostic_60_selftest.csv
```

Self-test memakai command STCC4 `0x278C`, menunggu 360 ms, lalu memvalidasi respons 16-bit dan CRC sesuai datasheet Sensirion. Opsi ini tidak dijalankan secara default. Jika proses dihentikan dengan `Ctrl+C`, baris yang sudah tertulis tetap tersimpan dan koneksi I2C ditutup.

Interpretasi hasil:

- `crc_all_valid=True` berarti keempat word CO2, temperatur, kelembapan, dan status memiliki CRC yang benar.
- `status_msb=0x01` dan `status_lsb=0x00` menghasilkan `status_decimal=256` (`0x0100`).
- `status_transition` dan ringkasan terminal menunjukkan perubahan antara `0x0000` dan `0x0100`.
- Data berstatus `0x0100` tetap diagnostik dan tidak otomatis dianggap valid untuk baseline; skrip tidak mengubah aturan validasi produksi.
- Baris `measurement_available=False` atau CRC invalid harus diperiksa sebagai masalah komunikasi/data sebelum digunakan untuk penelitian.

Setelah selesai, tinjau CSV tersebut sebelum training:

```bash
python3 -m src.offline_training \
  data/baseline_sensor_normal.csv \
  --model data/models/isolation_forest.joblib \
  --review
```

## Prosedur pengumpulan baseline normal

1. Jalankan sensor pada kondisi kabin yang dinyatakan normal dan stabil, tanpa kejadian gas yang sengaja dibuat.
2. Tunggu warm-up sensor dan catat sekurang-kurangnya beberapa puluh menit dengan interval target 1 detik. Dokumentasikan waktu, kondisi kabin, ventilasi, temperatur, kelembapan, dan status sensor.
3. Simpan CSV dengan kolom minimal `timestamp`, `co_ppm`, dan `co2_ppm`. Kolom `source=SENSOR` dan catatan kondisi pengukuran disarankan.
4. Tinjau statistik, timestamp duplikat, nilai invalid, status sensor, dan indikasi kejadian abnormal:

```powershell
python -m src.offline_training baseline_sensor_normal.csv --model data/models/isolation_forest.joblib --review
```

5. Hanya setelah dataset disetujui secara metodologis, jalankan `--approve`. Dataset testing, data threshold crossing, spike, dan data abnormal tidak boleh berada dalam baseline training.

## Pengujian

```powershell
python -m pytest -q -p no:cacheprovider
```

Training selalu menggunakan data normal terpisah. Suhu dan kelembapan disimpan ke CSV/dashboard, tetapi bukan fitur Isolation Forest. Driver STCC4 dipertahankan sesuai atribusi DFRobot dan MIT License.
