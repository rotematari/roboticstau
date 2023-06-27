# roboticstau
FMG_Project
![img.png](img.png)

lower arm
#define S11 A16  //pin 40
#define S9 A2   //pin 16
#define S8 A1   //pin 15
upper arm 
#define S10 A3   //pin 17
#define S7 A15  //pin 39

sholder 
#define S6 A14  //pin 38
#define S5 A13  //pin 27
#define S4 A12  //pin 26
#define S3 A11  //pin 25
#define S2 A10  //pin 24
#define S1 A9   //pin 23


comand line for teensy 4.1 dui 

wget https://downloads.arduino.cc/arduino-1.8.15-linux64.tar.xz
wget https://www.pjrc.com/teensy/td_154/TeensyduinoInstall.linux64
wget https://www.pjrc.com/teensy/00-teensy.rules
sudo cp 00-teensy.rules /etc/udev/rules.d/
tar -xf arduino-1.8.15-linux64.tar.xz
chmod 755 TeensyduinoInstall.linux64
./TeensyduinoInstall.linux64 --dir=arduino-1.8.15
cd arduino-1.8.15/hardware/teensy/avr/cores/teensy4
make