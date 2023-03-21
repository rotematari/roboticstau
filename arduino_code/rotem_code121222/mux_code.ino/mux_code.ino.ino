


#define A13 A13  
#define A12 A12  
#define A11 A11 


//16-Channel MUX (74HC4067) Interface
//===================================
int i,j; 
int S[4] = {9,10,11,12};
int A[3] = {A11,A12,A13};
int MUXtable[16][4]=
{
  {0,0,0,0}, {1,0,0,0}, {0,1,0,0}, {1,1,0,0},
  {0,0,1,0}, {1,0,1,0}, {0,1,1,0}, {1,1,1,0},
  {0,0,0,1}, {1,0,0,1}, {0,1,0,1}, {1,1,0,1},
  {0,0,1,1}, {1,0,1,1}, {0,1,1,1}, {1,1,1,1}
};

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
  for(j=0; j<3; i++)
  {
    for(i=0; i<2; i++)
    {
//      Serial.print("#####");Serial.println(i);
      selection(i);
      delay(500);
      Serial.print(analogRead(A[0]));Serial.println(",");
      
      
    }
  }
}
//=================================================
void selection(int j)
{
  digitalWrite(S[0], MUXtable[j][0]);
  digitalWrite(S[1], MUXtable[j][1]);
  digitalWrite(S[2], MUXtable[j][2]);
  digitalWrite(S[3], MUXtable[j][3]);

}
