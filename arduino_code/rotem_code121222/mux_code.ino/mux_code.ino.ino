


#define A12 A12  
#define A11 A11  
#define A10 A10 


//16-Channel MUX (74HC4067) Interface
//===================================
int i,j,count,a_read ; 
int S[4] = {9,10,11,12};
int A[3] = {A10,A11,A12};
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
analogReadResolution(12);
}
//=================================================
void loop()
{
  
  for(j=0; j<3; j++)
  {
    for(i=0; i<16; i++)
    {
      selection(i);
      delayMicroseconds(100);
      a_read =  analogRead(A[j]) ;
      if (a_read>=30 &&a_read<1800)
      {
        count  = j*16+i ; 
        Serial.print(count);Serial.print(":");Serial.print(a_read);Serial.print(",");
        
//        Serial.print("count = ");Serial.println(count);
        }
      
      
      
    }
//    delay(200);
    
  }
  Serial.println("");
}
//=================================================
void selection(int j)
{
  digitalWrite(S[0], MUXtable[j][0]);
  digitalWrite(S[1], MUXtable[j][1]);
  digitalWrite(S[2], MUXtable[j][2]);
  digitalWrite(S[3], MUXtable[j][3]);

}
