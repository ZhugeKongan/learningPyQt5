#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""Contains adapter implementations for MAX78000 EvKit to get CNN model output.
"""

from collections import Counter
import time
import numpy as np
import serial


class AI85Adapter:
    """
    Adapter base class for MAX78000 devices to get network output.
    """
    simulator = None
    def __init__(self):
        pass

    def get_network_out(self, data):
        """Returns output of the neural network on device."""

    def __del__(self):
        pass


class AI85SimulatorAdapter(AI85Adapter):
    """
    Adapter for MAX78000 simulator.
    """
    def __init__(self, path_to_checkpoint): #pylint: disable=unused-argument
        super().__init__()

    def get_network_out(self, data):
        """Returns output of the neural network on device."""
        print('Not implemented yet!!')

    def __del__(self):
        pass


class AI85SpiAdapter(AI85Adapter):
    """
    Adapter for MAX78000 SPI interface.
    """
    def get_network_out(self, data):
        """Returns output of the neural network on device."""
        print('Not implemented yet!!')

    def __del__(self):
        pass


class AI85UartAdapter(AI85Adapter):
    """
    Adapter for MAX78000 UART interface.
    """
    run_test_code = bytes([58])

    def __init__(self, port, baud_rate, embedding_len):
        super().__init__()
        self.ser = serial.Serial(port=port, baudrate=baud_rate, bytesize=8, parity='N', stopbits=1,
                                 timeout=0, xonxoff=0, rtscts=0)
        self.embedding_len = embedding_len

    def get_network_out(self, data):
        """Returns output of the neural network on device."""
        self.ser.flushInput()#清空缓冲区
        self.__connect_device()#链接设备
        print('Connected')
        self.__transmit_data(data)#开始数据传输
        time.sleep(1)
        print('Transmitted')
        model_out = self.__receive_data()#从串口接收模型输出
        self.__release_device()
        return model_out

    def __del__(self):
        pass

    def __connect_device(self):
        initial_device_message = b'Start_Sequence'
        print('Listening to connect device!')

        while True:
            sync_message = self.ser.read(len(initial_device_message))#读取串口，循环等待初始化字段
            print(sync_message)
            if self.__check_synch_message(sync_message, initial_device_message):
                break
            time.sleep(0.05)
        print('From Device: %s' % sync_message)

        self.ser.write(self.run_test_code)#串口写入测试信号/58
        time.sleep(1)

    def __check_synch_message(self, sync_message, initial_device_message): #pylint: disable=no-self-use
        if Counter(sync_message) == Counter(initial_device_message):
            return True
        return False

    def __release_device(self):
        final_device_message = b'End_Sequence'
        print('Listening to release device!')

        while True:
            sync_message = self.ser.read(len(final_device_message))
            print(sync_message)
            if sync_message == final_device_message:
                self.ser.write(bytes([100]))
                break
            time.sleep(1)
        print('From Device: %s' % sync_message)

    def __transmit_data(self, data):
        temp = np.int8(data - 128)
        byte_arr = temp.copy()
        byte_arr[:, :, 0] = temp[:, :, 2]
        byte_arr[:, :, 2] = temp[:, :, 0]
        byte_arr = byte_arr.tobytes()
        self.ser.write(byte_arr)

    def __receive_data(self):
        size = self.embedding_len
        array = b""
        while size > 0:
            len_to_read = self.ser.inWaiting()
            if size < len_to_read:
                len_to_read = size
            size = size - len_to_read
            array += self.ser.read(len_to_read)
        return np.frombuffer(array, np.int8)
