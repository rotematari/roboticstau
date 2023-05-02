
#define A13 A12
int a_read=0;


void setup() {
  Serial.begin(9600);
  analogReadResolution(12);

}

void loop() {
  a_read =  analogRead(A13) ;
  Serial.println(a_read);
  delay(100);

}
