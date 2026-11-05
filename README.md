# Purwarupa Deteksi Anomali Gas CO dan CO₂

Repository ini memisahkan akuisisi, pemrosesan, model, dan dashboard. Sensor aktif adalah Winsen ZE07-CO melalui UART dan DFRobot Gravity STCC4 SEN0678 melalui I²C. SCD41 tidak digunakan oleh alur aktif.

## Arsitektur

- `src/sensors/`: adapter ZE07-CO dan STCC4. Adapter tidak membuka hardware ketika modul diimpor.
- `src/acquisition_service.py`: proses terpisah yang membaca DUMMY atau SENSOR setiap 1 detik, memvalidasi status, menjalankan SMA causal 5 sampel, inference Isolation Forest, dan menulis CSV.
- `src/sensor_pipeline.py`: pipeline bersama untuk validasi → SMA → fitur `co_ma`, `co2_ma` → inference → warning.
- `src/dashboard.py`: Streamlit read-only terhadap CSV; browser refresh tidak menginisialisasi sensor atau melatih model.
- `src/sensors/vendor/DFRobot_STCC4.py`: kode driver DFRobot resmi yang divendorkan dengan atribusi dan lisensi MIT.

Threshold 35 ppm CO dan 1000 ppm CO₂ adalah parameter simulasi operasional, bukan klaim batas kesehatan sesaat. Threshold alert memakai nilai sensor valid sebelum SMA. Data invalid, termasuk status STCC4 256/0x0100 yang belum terverifikasi pada unit fisik, tidak masuk SMA atau model.

## Mode simulasi Windows

```powershell
python -m pip install -r requirements.txt
python -m src.acquisition_service --mode dummy --scenario normal
```

Di terminal kedua:

```powershell
python -m streamlit run src/dashboard.py --server.port 8501
```

Buka `http://localhost:8501`. Skenario dapat dipilih melalui kontrol dashboard; tombol Start/Pause/Reset mengirim perintah ke acquisition service melalui file kontrol. Untuk smoke test tanpa menunggu lama:

```powershell
python -m src.acquisition_service --mode dummy --scenario sudden_spike --max-samples 8 --output data/smoke.csv --model data/models/smoke.joblib
```

## Deployment Raspberry Pi 4

Aktifkan I²C, pastikan ZE07-CO terhubung ke UART `/dev/serial0` pada 9600 baud, dan verifikasi STCC4 pada bus 1/alamat `0x64` (`i2cdetect -y 1`). Install dependensi dan jalankan acquisition sebagai proses terpisah:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.acquisition_service --mode sensor
```

Untuk akses dashboard dari laptop/HP pada jaringan yang sama:

```bash
python3 -m streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Kemudian buka `http://ALAMAT_IP_RASPBERRY_PI:8501`. Jangan menjalankan `--mode sensor` di Windows.

## Pengujian

```powershell
python -m pytest -q -p no:cacheprovider
```

Training selalu menggunakan data normal terpisah. Suhu dan kelembapan disimpan ke CSV/dashboard, tetapi bukan fitur Isolation Forest. Driver STCC4 dipertahankan sesuai atribusi DFRobot dan MIT License.
