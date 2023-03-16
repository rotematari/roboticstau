
//16-Channel MUX (74HC4067) Interface
//===================================
int i; 
int S[4] = {0,1,2,3};
int MUXtable[16][4]=
{
  {0,0,0,0}, {1,0,0,0}, {0,1,0,0}, {1,1,0,0},
  {0,0,1,0}, {1,0,1,0}, {0,1,1,0}, {1,1,1,0},
  {0,0,0,1}, {1,0,0,1}, {0,1,0,1}, {1,1,0,1},
  {0,0,1,1}, {1,0,1,1}, {0,1,1,1}, {1,1,1,1}
};
//int MUXtable[16][4]=
//{
//  {LOW,LOW,LOW,LOW}, {HIGH,LOW,LOW,LOW}, {LOW,HIGH,LOW,LOW}, {HIGH,HIGH,LOW,LOW},
//  {LOW,LOW,HIGH,LOW}, {HIGH,LOW,HIGH,LOW}, {LOW,HIGH,HIGH,LOW}, {HIGH,HIGH,HIGH,LOW},
//  {LOW,LOW,LOW,HIGH}, {HIGH,LOW,LOW,HIGH}, {LOW,HIGH,LOW,HIGH}, {HIGH,HIGH,LOW,HIGH},
//  {LOW,LOW,HIGH,HIGH}, {HIGH,LOW,HIGH,HIGH}, {LOW,HIGH,HIGH,HIGH}, {HIGH,HIGH,HIGH,HIGH}
//};
//=================================================



//void loop() {
//
//  
//  
//  Serial.print(analogRead(A0));
//  Serial.println(",");
//  delay(300);
//
//}


void setup()
{
  for(i=0; i<4; i++){ 
    pinMode(S[i],OUTPUT);  
    digitalWrite(S[i], LOW);
  }
Serial.begin(9600);
}
//=================================================
void loop()
{
  for(i=13; i<16; i++)
  {
    Serial.print("#####");Serial.println(i);
    selection(i);
    delay(1000);
    Serial.print(analogRead(A0));
    Serial.println(",");
    
  }
}
//=================================================
void selection(int j)
{
  digitalWrite(S[0], MUXtable[j][0]);
  digitalWrite(S[1], MUXtable[j][1]);
  digitalWrite(S[2], MUXtable[j][2]);
  digitalWrite(S[3], MUXtable[j][3]);
//  digitalWrite(S[0], LOW);
//  digitalWrite(S[1], HIGH);
//  digitalWrite(S[2], HIGH);
//  digitalWrite(S[3], HIGH);
  
//  Serial.print(S[0]);Serial.println(MUXtable[j][0]);
//  Serial.print(S[1]);Serial.println(MUXtable[j][1]);
//  Serial.print(S[2]);Serial.println(MUXtable[j][2]);
//  Serial.print(S[3]);Serial.println(MUXtable[j][3]);
}
