

//Motor Definitions

int E1 = 5;     //M1 Speed Control
int E2 = 6;     //M2 Speed Control
int M1 = 4;     //M1 Direction Control
int M2 = 7;     //M2 Direction Control

//DIRECTIONS

//STOP
void stop(void)
{
  digitalWrite(E1, 0);
  digitalWrite(M1, LOW);
  digitalWrite(E2, 0);
  digitalWrite(M2, LOW);
}

//ADVANCE
void advance(char a, char b)
{
  analogWrite (E1, a);
  digitalWrite(M1, HIGH);
  analogWrite (E2, b);
  digitalWrite(M2, HIGH);
}

//MOVE BACKWARDS
void back_off (char a, char b)
{
  analogWrite (E1, a);
  digitalWrite(M1, LOW);
  analogWrite (E2, b);
  digitalWrite(M2, LOW);
}


//TURN LEFT
void turn_L (char a, char b)
{
  analogWrite (E1, a);
  digitalWrite(M1, LOW);
  analogWrite (E2, b);
  digitalWrite(M2, HIGH);
}

//TURN RIGHT
void turn_R (char a, char b)
{
  analogWrite (E1, a);
  digitalWrite(M1, HIGH);
  analogWrite (E2, b);
  digitalWrite(M2, LOW);
}

void setup(void) {
  char val ;

  int i;
  for (i = 4; i <= 7; i++)
    pinMode(i, OUTPUT);
  Serial.begin(9600);      
  Serial.println("hello. w = forward, d = turn right, a = turn left, s = backward, x = stop, z = hello world"); //Display instructions in the serial monitor
  digitalWrite(E1, LOW);
  digitalWrite(E2, LOW);
}

void loop(void) {
  if ( Serial.available() ) {
    char  val = Serial.read();
    if (val != -1) 
    {
      switch (val)
      {
        case '5'://Move backward
          Serial.println("backward");
          turn_L (70, 70);
          delay (1000);
          stop();
          break;
        case '8'://Move forward
          Serial.println("forward");
          turn_R (70, 70);
          delay (1000);
          stop();
          break;
         
        case '4'://Turn Left
          Serial.println("going forward");
          advance (95, 95);  //move forward at max speed
          delay (2500);
          stop();
          break;
        case '6'://Turn Right
          Serial.println("going backward");
          back_off (95, 95);  //move backwards at max speed
          delay (2500);
          stop();
          break;
        case '1' : //turn right "doucement"
           Serial.println("going backward");
           back_off (90, 90);  //move backwards at max speed
           delay (900);
           stop();
           break;
        case '2' : //turn left "doucement"
           Serial.println("going forward");
           advance (90, 90);  //move forward at max speed
           delay (900);
           stop();
           break;
        case '3' : //backword "doucement"
           Serial.println("going forward");
           turn_L (55, 55);  //move forward at max speed
           delay (1000);
           stop();
           break;
        case '7' : //avnace "doucement"
           Serial.println("going forward");
           turn_R (55, 55);  //move forward at max speed
           delay (1000);
           stop();
           break;
           
        default :
          break;
      }
  }
  }
  else stop();
}
