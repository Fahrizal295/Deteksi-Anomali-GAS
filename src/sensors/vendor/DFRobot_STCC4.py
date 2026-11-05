"""!
    * @file DFRobot_STCC4.py
    * @brief Declaration the basic structure of class DFRobot_STCC4
    * @n It is possible to measure the concentration of carbon dioxide in the air.
    * @n If the humidity and temperature sensor is connected, temperature and humidity can also be obtained.
    * @n Only supports IIC communication interface
    * @copyright	Copyright (c) 2025 DFRobot Co.Ltd (http://www.dfrobot.com)
    * @license The MIT License (MIT)
    * @author [fary](feng.yang@dfrobot.com)
    * @version V1.0
    * @date 2025-08-15
    * @url https://github.com/DFRobot/DFRobot_STCC4
 """

import smbus2
import time
from typing import Optional, Tuple, Union
 
class DFRobot_STCC4:
    """Base class for DFRobot STCC4 CO2 sensor"""
    
    ERR_OK = 0
    ERR_DATA_BUS = 1
    ERR_DATA_READ = 2
    ERR_DATA_WRITE = 3
    ERR_IC_VERSION = 4
    
    # Sensor commands
    STCC4_GET_ID = 0x365B
    STCC4_START_CONT_MEASURE = 0x218B
    STCC4_STOP_CONT_MEASURE = 0x3F86
    STCC4_READ_MEASURE = 0xEC05
    STCC4_SET_RHT_COMPENSATION = 0xE000
    STCC4_SET_PRESSURE_COMPENSATION = 0xE016
    STCC4_SINGLE_SHOT = 0x219D
    STCC4_SLEEP = 0x3650
    STCC4_WAKEUP = 0x00
    STCC4_SOFT_RESET = 0x06
    STCC4_FACTORY_RESET = 0x3632
    STCC4_ENABLE_TESTING_MODE = 0x3FBC
    STCC4_DISABLE_TESTING_MODE = 0x3F3D
    STCC4_FORC_CALIBRATION = 0x362F
 
    def __init__(self):
        """Constructor"""
        pass
 
    def calculation_crc(self, data: Union[list, tuple]) -> int:
        """
        Calculate CRC for the data
        :param data: List or tuple of 16-bit integers
        :return: Calculated CRC value
        """
        crc = 0xFF
        
        for value in data:
            high_byte = (value >> 8) & 0xFF
            low_byte = value & 0xFF
            
            # Process high byte
            crc ^= high_byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x31
                else:
                    crc <<= 1
                crc &= 0xFF  # Keep it as 8-bit
            
            # Process low byte
            crc ^= low_byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x31
                else:
                    crc <<= 1
                crc &= 0xFF  # Keep it as 8-bit
        
        return crc
 
    def get_id(self) -> Optional[bytes]:
        """
        Get the sensor ID
        :return: Sensor ID as bytes if successful, None otherwise
        """
        raise NotImplementedError
 
    def start_measurement(self) -> bool:
        """
        Start continuous measurement
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def stop_measurement(self) -> bool:
        """
        Stop continuous measurement
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def measurement(self) -> Optional[Tuple[int, float, float, int]]:
        """
        Read measurement data
        :return: Tuple of (co2_concentration, temperature, humidity, sensor_status).
        co2_concentration : CO2 concentration
        temperature : Temperature
        humidity : Humidity
        sensor_status : Sensor status
        None : error
        """
        raise NotImplementedError
 
    def set_rht_compensation(self, temperature: int, humidity: int) -> bool:
        """
        Set temperature and humidity compensation
        :param temperature: Temperature compensation value, range of 10 to 40 degrees Celsius.
        :param humidity: Humidity compensation value, range of 20 to 80 percent relative humidity.
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def set_pressure_compensation(self, pressure: int) -> bool:
        """
        Set pressure compensation
        :param pressure: Pressure compensation value, range of 400 to 1100 hPa
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def single_measurement(self) -> bool:
        """
        Perform a single shot measurement
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def fall_asleep(self) -> bool:
        """
        Put the sensor into sleep mode
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def wakeup(self) -> bool:
        """
        Wake up the sensor from sleep mode
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def soft_reset(self) -> bool:
        """
        Perform a soft reset of the sensor
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def factory_reset(self) -> bool:
        """
        Perform a factory reset of the sensor
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def enable_testing_mode(self) -> bool:
        """
        Enable testing mode
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def disable_testing_mode(self) -> bool:
        """
        Disable testing mode
        :return: True if successful, False otherwise
        """
        raise NotImplementedError
 
    def forced_recalibration(self, target_ppm: int) -> Optional[int]:
        """
        Perform forced recalibration
        :param target_ppm: Target PPM value for recalibration, must be between 0 and 32000 ppm.
        :return: Correction value if successful, None otherwise
        """
        raise NotImplementedError
 
 
class DFRobot_STCC4_I2C(DFRobot_STCC4):
    """I2C implementation of DFRobot STCC4 CO2 sensor"""
    
    DEFAULT_I2C_ADDR = 0x64
    I2C_BUS = 1  # Raspberry Pi uses bus 1 for I2C
 
    def __init__(self, addr: int = DEFAULT_I2C_ADDR):
        """
        Constructor for I2C implementation
        :param addr: I2C address of the sensor
        """
        super().__init__()
        self._device_addr = addr
        try:
            self._bus = smbus2.SMBus(self.I2C_BUS)
        except Exception as e:
            self._bus = None
 
    def _write_cmd16(self, cmd: int) -> bool:
        """
        Write a 16-bit command to the sensor
        :param cmd: Command to write
        :return: True if successful, False otherwise
        """
        if self._bus is None:
            return False
            
        try:
            # Split 16-bit command into two bytes (big endian)
            bytes_to_send = [(cmd >> 8) & 0xFF, cmd & 0xFF]
            self._bus.write_i2c_block_data(self._device_addr, bytes_to_send[0], [bytes_to_send[1]])
            return True
        except Exception as e:
            return False
 
    def _write_cmd8(self, cmd: int) -> bool:
        """
        Write an 8-bit command to the sensor
        :param cmd: Command to write
        :return: True if successful, False otherwise
        """
        if self._bus is None:
            return False
            
        try:
            self._bus.write_byte(self._device_addr, cmd)
            return True
        except Exception as e:
            return False
 
    def _write_data(self, cmd: int, data: Union[list, tuple]) -> bool:
        """
        Write data to the sensor
        :param cmd: Command to write before data
        :param data: List or tuple of 16-bit integers to write
        :return: True if successful, False otherwise
        """
        if not self._write_cmd16(cmd):
            return False
            
        try:
            for value in data:
                # Split value into two bytes
                high_byte = (value >> 8) & 0xFF
                low_byte = value & 0xFF
                
                # Calculate CRC for this value
                crc = self.calculation_crc([value])
                
                # Write data bytes and CRC

                # self._bus.write_i2c_block_data(
                #     self._device_addr, 
                #     high_byte, 
                #     [low_byte, crc]
                # )
                self._bus.write_byte(self._device_addr, high_byte)
                self._bus.write_byte(self._device_addr, low_byte)
                self._bus.write_byte(self._device_addr, crc)
                # Small delay between writes
                time.sleep(0.01)
            
            return True
        except Exception as e:
            return False
 
    def _read_data(self, cmd: int, length: int) -> Optional[bytes]:
        """
        Read data from the sensor
        :param cmd: Command to write before reading
        :param length: Number of bytes to read
        :return: Read bytes if successful, None otherwise
        """
        if not self._write_cmd16(cmd):
            return None
            
        try:
            # Read the data
            data = self._bus.read_i2c_block_data(self._device_addr, 0, length)
            return bytes(data)
        except Exception as e:
            return None
 
    def get_id(self):
        """
        Get sensor ID with CRC check
        Returns:
            int: 32-bit sensor ID
        """
        for i in range(5):
            r_buf = self._read_data(self.STCC4_GET_ID, 18)
            if r_buf is None or len(r_buf) < 18:
                time.sleep(0.2)
                continue
                
            # Extract data pairs and their CRCs
            data1 = (r_buf[0] << 8) | r_buf[1]
            data2 = (r_buf[3] << 8) | r_buf[4]
            crc1 = r_buf[2]
            crc2 = r_buf[5]
            
            # Calculate CRCs
            calculated_crc1 = self.calculation_crc([data1])
            calculated_crc2 = self.calculation_crc([data2])
            
            # Check CRCs
            if crc1 == calculated_crc1 and crc2 == calculated_crc2:
                id_value = (r_buf[0] << 24) | (r_buf[1] << 16) | (r_buf[3] << 8) | r_buf[4]
                if id_value == 0x901018A:
                    return id_value
                    
            time.sleep(0.2)

        return 0
 
    def start_measurement(self) -> bool:
        """Start continuous measurement"""
        if not self._write_cmd16(self.STCC4_START_CONT_MEASURE):
            return False
        time.sleep(1)
        return True
 
    def stop_measurement(self) -> bool:
        """Stop continuous measurement"""
        if not self._write_cmd16(self.STCC4_STOP_CONT_MEASURE):
            return False
        time.sleep(1)
        return True
 
    def measurement(self) -> Optional[Tuple[int, float, float, int]]:
        """Read measurement data"""
        raw_data = self._read_data(self.STCC4_READ_MEASURE, 12)
        if raw_data is None or len(raw_data) < 12:
            return None
            
        # Parse CO2 concentration
        co2_concentration = (raw_data[0] << 8) | raw_data[1]
        
        # Parse temperature (raw value to °C)
        temp_raw = (raw_data[3] << 8) | raw_data[4]
        temperature = -45.0 + ((175.0 * temp_raw) / 65535.0)
        
        # Parse humidity (raw value to %RH)
        hum_raw = (raw_data[6] << 8) | raw_data[7]
        humidity = -6.0 + ((125.0 * hum_raw) / 65535.0)
        
        # Parse sensor status
        sensor_status = (raw_data[9] << 8) | raw_data[10]
        
        return (co2_concentration, temperature, humidity, sensor_status)
 
    def set_rht_compensation(self, temperature: float, humidity: float) -> bool:
        """Set temperature and humidity compensation"""
        if temperature < 10 or temperature > 40 or humidity < 20 or humidity > 80:
            return False
        # Convert temperature to raw value
        temp_raw = int((temperature + 45) * 65535 / 175)
        # Convert humidity to raw value
        hum_raw = int((humidity + 6) * 65535 / 125)
        
        return self._write_data(self.STCC4_SET_RHT_COMPENSATION, [temp_raw, hum_raw])
 
    def set_pressure_compensation(self, pressure: int) -> bool:
        """Set pressure compensation"""
        if pressure < 400 or pressure > 1100:
            return False
        pressure_raw = pressure * 50
        return self._write_data(self.STCC4_SET_PRESSURE_COMPENSATION, [pressure_raw])
 
    def single_measurement(self) -> bool:
        """Perform single shot measurement"""
        return self._write_cmd16(self.STCC4_SINGLE_SHOT)

    def fall_asleep(self) -> bool:
        """Put sensor to sleep"""
        return self._write_cmd16(self.STCC4_SLEEP)
 
    def wakeup(self) -> bool:
        """Wake up sensor"""
        return self._write_cmd8(self.STCC4_WAKEUP)
 
    def soft_reset(self) -> bool:
        """Perform soft reset"""
        return self._write_cmd8(self.STCC4_SOFT_RESET)
 
    def factory_reset(self) -> bool:
        """Perform factory reset"""
        if not self._write_cmd16(self.STCC4_FACTORY_RESET):
            return False
            
        raw_data = self._read_data(self.STCC4_FACTORY_RESET, 2)
        if raw_data is None or len(raw_data) < 2:
            return False
            
        response = (raw_data[0] << 8) | raw_data[1]
        return response == 0
 
    def enable_testing_mode(self) -> bool:
        """Enable testing mode"""
        return self._write_cmd16(self.STCC4_ENABLE_TESTING_MODE)
 
    def disable_testing_mode(self) -> bool:
        """Disable testing mode"""
        return self._write_cmd16(self.STCC4_DISABLE_TESTING_MODE)
 
    def forced_recalibration(self, target_ppm: int) -> Optional[int]:
        """Perform forced recalibration"""
        if  target_ppm > 32000:
            return None
        if not self._write_data(self.STCC4_FORC_CALIBRATION, [target_ppm]):
            return None
            
        time.sleep(0.2)  # 200ms delay
        
        raw_data = self._read_data(self.STCC4_FORC_CALIBRATION, 3)
        if raw_data is None or len(raw_data) < 3:
            return None
            
        frc_correction = (raw_data[0] << 8) | raw_data[1]
        return frc_correction