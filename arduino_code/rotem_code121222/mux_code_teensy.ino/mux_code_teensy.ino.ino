
#define A12 A12
#define A11 A11
#define A10 A10
#define A9 A9

#define A7 A7
#define A6 A6
#define A5 A5
#define A4 A4


//16-Channel MUX (74HC4067) Interface
//===================================
int i, j, count ;
float a_read;
int S[4] = {9,10,11,12};
int A[3] = {A10,A12,A12};
int MUXtable[16][4] =
{
  {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0},
  {0, 0, 1, 0}, {1, 0, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0},
  {0, 0, 0, 1}, {1, 0, 0, 1}, {0, 1, 0, 1}, {1, 1, 0, 1},
  {0, 0, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}
};

//=================================================




void setup()
{
  for (i = 0; i < 4; i++) {
    pinMode(S[i], OUTPUT);
    digitalWrite(S[i], LOW);

  }
  Serial.begin(115200);
  analogReadResolution(12);
}
//=================================================
void loop()
{
   
    for(i=0; i<16; i++)
    {
       
       selection(i);
       delay(5);
       for(j=0; j<2; j++)
    {
      
      a_read =  analogRead(A[j]) ;
      Serial.print(a_read);Serial.print(",");
        }
    
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
