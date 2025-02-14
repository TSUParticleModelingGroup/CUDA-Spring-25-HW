// nvcc OpenGLTest.cu -o temp -lglut -lm -lGLU -lGL

#include <GL/glut.h>
#include <stdio.h>
#include <GL/gl.h>

#define XWINDOWSIZE 500
#define YWINDOWSIZE 500

// Window globals
static int Window;
int XWindowSize;
int YWindowSize; 
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;

// Prototyping functions
void display();
void idle();
void reshape(int, int);
void keyPressed(unsigned char, int, int);
void mymouse(int, int, int, int);
void drawPicture();
int main(int, char**);

void display()
{
	printf("Display----\n");
}

void idle()
{
	//printf("idle\n");
}

void reshape(int w, int h)
{
	printf("Reshape----\n");
}

void keyPressed(unsigned char key, int x, int y)
{	

	printf("Key Press----\n");
	
	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
	
	if(key == 'p')
	{
		printf("\nYou just pressed the p key.\n");
	}
	
}

void mousePassiveMotionCallback(int x, int y) 
{
	// This function is called when the mouse moves without any button pressed
	// x and y are the current mouse coordinates
	
	// x and y come in as 0 to XWindowSize and 0 to YWindowSize. This traveslates them to -1 to 1 and -1 to 1.
	printf("\n Mouse motion");
	printf("\n mouse x = %d mouse y = %d\n", x, y);
	
	//drawPicture();

	//printf("\n MouseX = %f\n", MouseX);
}

void mymouse(int button, int state, int x, int y)
{	
	// x and y come in as 0 to XWindowSize and 0 to YWindowSize. This traveslates them to -1 to 1 and -1 to 1.
	printf("\n Mouse buttons");
	if(state == GLUT_DOWN)
	{
		if(button == GLUT_LEFT_BUTTON)
		{
			printf("\n Left mouse button down");
			printf("\n mouse x = %d mouse y = %d\n", x, y);
		}
		
		if(button == GLUT_RIGHT_BUTTON)
		{
			printf("\n Right mouse button down");
			printf("\n mouse x = %d mouse y = %d\n", x, y);
		}
		
		if(button == GLUT_MIDDLE_BUTTON)
		{
			printf("\n Middle mouse button down");
			printf("\n mouse x = %d mouse y = %d\n", x, y);
		}
	}
	
	if(state == 0)
	{
		if(button == 3)
		{
			printf("\n Scrolling in");
		}
		else if(button == 4)
		{
			printf("\n Scrolling out");
		}
	}
}

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0, 1.0, 0.0);
	glPushMatrix();
		//glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
		glutSolidSphere(1.0, 30, 30);
	glPopMatrix();

	glutSwapBuffers();
}
                                 
int main(int argc, char** argv)
{
	XWindowSize = XWINDOWSIZE;
	YWindowSize = YWINDOWSIZE; 

	// Clip planes
	Near = 0.2;
	Far = 30.0;

	// Eye position
	EyeX = 0.0;
	EyeY = 0.0;
	EyeZ = 5.0;

	// Center position
	CenterX = 0.0;
	CenterY = 0.0;
	CenterZ = 0.0;

	// Up vector
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize, YWindowSize);
	glutInitWindowPosition(5, 5);
	Window = glutCreateWindow("OpenGL Test");

	glViewport(0, 0, XWindowSize, YWindowSize);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double)XWindowSize / (double)YWindowSize, Near, Far); // Use gluPerspective for better 3D projection
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);

	glClearColor(0.0, 0.0, 0.0, 0.0);

	GLfloat light_position2[] = {1.0, 1.0, 0.0, 0.0};
	GLfloat light_position[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};

	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	glLightfv(GL_LIGHT0, GL_POSITION, light_position2);
	glEnable(GL_LIGHT1);

	drawPicture();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutPassiveMotionFunc(mousePassiveMotionCallback);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(keyPressed);
	glutIdleFunc(idle);
	glutMainLoop();
	return 0;
}




