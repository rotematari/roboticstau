
// Call of libraries
#include <Wire.h>
#include <SparkFunLSM9DS1.h>
#include <SoftwareSerial.h>
#include <XBee.h>



#define S11 A16  //pin 40
#define S10 A3   //pin 17
#define S9 A2   //pin 16
#define S8 A1   //pin 15
#define S7 A15  //pin 39
#define S6 A14  //pin 38
#define S5 A13  //pin 27
#define S4 A12  //pin 26
#define S3 A11  //pin 25
#define S2 A10  //pin 24
#define S1 A9   //pin 23




//SET XBEE3_PRO COMMUNICTION
const byte rxPin = 0;
const byte txPin = 1;
SoftwareSerial hc06 (rxPin, txPin);//define RF COM


// defining module addresses
#define LSM9DS1_M  0x1E //magnometer
#define LSM9DS1_AG 0x6B //accelometer and gyroscope
LSM9DS1 imu; // Creation of the object
const int buttonPin = 2;
int button = false;
int i = 1;
unsigned long myTime0 = millis();
int baudRate = 115200;

#define DECLINATION -5.55 //declination in degrees in tel-aviv
#define PRINT_CALCULATED

void setup()
{
  //XBEE Pins
  pinMode(rxPin, INPUT);
  pinMode(txPin, OUTPUT);
  pinMode(buttonPin, INPUT);
  pinMode(31, OUTPUT);
  pinMode(30, OUTPUT);
  digitalWrite(30, LOW);
  digitalWrite(31, LOW);

  //  attachInterrupt(digitalPinToInterrupt(buttonPin), raise_flag, RISING);

  //IMU init
  Serial.begin(baudRate);//initialization of  SERIAL 9600 B.R
  Wire.begin();     //initialization of the I2C communication  //PRINT IMU DATA
  imu.settings.device.commInterface = IMU_MODE_I2C;
  imu.settings.device.mAddress = LSM9DS1_M;
  imu.settings.device.agAddress = LSM9DS1_AG;
  if (!imu.begin()) { //display error message if that's the case
    hc06.println("Communication problem.");
    while (1);
  }

  //  Serial.println("SETUP:");
  hc06.begin(baudRate);//initialization of  xbee  SERIAL communication B.R
  //  Serial.println("begin:");
  //DEFINE I2C -IMU
  //Wire.begin(18, 19);
  Wire.setSCL(19);//DEFINE LEGS- SCL
  Wire.setSDA(18);//DEFINE LEG-SDA
  Wire.setClock(74880);
}

void loop()
{
  unsigned long myTime;
//  delay(100);

  digitalWrite(30, HIGH);
  digitalWrite(31, LOW);
  if (button == false) {
    if ( imu.gyroAvailable() ) {
      imu.readGyro(); //measure with the gyroscope
    }
    if ( imu.accelAvailable() ) {
      imu.readAccel(); //measure with the accelerometer
    }
    if ( imu.magAvailable() ) {
      imu.readMag(); //measure with the magnetometer
    }
    // print format: t,G,A,M,,F1,F2,F3,F4,B1,B2,B3,S1,S2
    myTime = millis();
    //    Serial.print("t,");
    Serial.print(String(float((myTime - myTime0)/1000.0)) + String(","));
    //PRINT IMU DATA
    printGyro();
    printAccel();
    printMag();
    //    delayMicroseconds(1600);


    //PRINT SENSORS DATA
    readset1();
    //    Serial.println();
    digitalWrite(31, HIGH);
    digitalWrite(30, LOW);
    delayMicroseconds(1600);
    //readset2();
    i = i + 1;


    //    Serial.println("num: " + i);

  }
  else {
    hc06.println(11111.0);
  }
}


void printGyro() {
  //  float array1[] = {1000001.0, imu.calcGyro(imu.gx), imu.calcGyro(imu.gy), imu.calcGyro(imu.gz)};
  Serial.print(String("") + imu.calcGyro(imu.gx) + String(",") + imu.calcGyro(imu.gy) + String(",") + imu.calcGyro(imu.gz) + String(","));
  //  Serial.println(String("G,") + imu.calcGyro(imu.gx) + String(",") + imu.calcGyro(imu.gy) + String(",") + imu.calcGyro(imu.gz));
  //  byte *p = (byte*)array1;
  //  for (byte i = 1; i <= sizeof(array1); i++){
  //    Serial.println("gyr" + i);
  //    Serial.print("G " + p[i]);
  //    hc06.println(p[i]);
  //    hc06.write(p[i]);
  //  }
}


void printAccel() {
  Serial.print(String("") + imu.calcAccel(imu.ax) + String(",") + imu.calcAccel(imu.ay) + String(",") + imu.calcAccel(imu.az) + String(","));
  //  Serial.println(String("A,") + imu.calcAccel(imu.ax) + String(",") + imu.calcAccel(imu.ay) + String(",") + imu.calcAccel(imu.az));
  //  float array2[] = {1000002.0, imu.calcAccel(imu.ax), imu.calcAccel(imu.ay), imu.calcAccel(imu.az)};
  //  byte *p = (byte*)array2;
  //  for (byte i = 0; i < sizeof(array2); i++){
  //  Serial.println("accel" + i);
  //  hc06.println(p[i]);
  //  Serial.println("A " + p[i]);
  //  hc06.write(p[i]);
  //  }
}


void printMag() {
  Serial.print(String("") + imu.calcMag(imu.mx) + String(",") + imu.calcMag(imu.my) + String(",") + imu.calcMag(imu.mz) + String(","));
  //  Serial.println(String("M,") + imu.calcMag(imu.mx) + String(",") + imu.calcMag(imu.my) + String(",") + imu.calcMag(imu.mz));
  //  float array3[] = {1000003.0, imu.calcMag(imu.mx), imu.calcMag(imu.my), imu.calcMag(imu.mz)};
  //  byte *p = (byte*)array3;
  //  for (byte i = 0; i < sizeof(array3); i++) {
  //    //Serial.println("A" + i);
  //    Serial.println("M " + p[i]);
  //    //hc06.println(p[i]);
  //    //hc06.write(p[i]);
  //  }
}


void readset1() {


  Serial.print(analogRead(S1)); Serial.print(",");
  Serial.print(analogRead(S2)); Serial.print(",");
  Serial.print(analogRead(S3)); Serial.print(",");
  Serial.print(analogRead(S4)); Serial.print(",");
  Serial.print(analogRead(S5)); Serial.print(",");
  Serial.print(analogRead(S6)); Serial.print(",");
  Serial.print(analogRead(S7)); Serial.print(",");
  Serial.print(analogRead(S8)); Serial.print(",");
  Serial.print(analogRead(S9)); Serial.print(",");
  Serial.print(analogRead(S10)); Serial.print(",");
  Serial.print(analogRead(S11)); Serial.println(",");
}


void readset2() {

}


void raise_flag() {
  button = true;
}
