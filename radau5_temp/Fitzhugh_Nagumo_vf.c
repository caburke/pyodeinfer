/*  Vector field function and events for Radau integrator.
  This code was automatically generated by PyDSTool, but may be modified by hand. */

#include <stdio.h>
#include "vfield.h"
#include <Python.h>
#include <math.h>
#include <string.h>
#include "signum.h"
#include <stdlib.h>
#include "maxmin.h"
#include "events.h"

extern double *gICs;
extern double **gBds;
extern double globalt0;

static double pi = 3.1415926535897931;

double signum(double x)
{
  if (x<0) {
    return -1;
  }
  else if (x==0) {
    return 0;
  }
  else if (x>0) {
    return 1;
  }
  else {
    /* must be that x is Not-a-Number */
    return x;
  }
}


/* Variable, aux variable, parameter, and input definitions: */ 
#define a	p_[0]
#define b	p_[1]
#define c	p_[2]
#define R	Y_[0]
#define V	Y_[1]


int getindex(char *name, double *p_, double *wk_, double *xv_);
double __maxof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_);
double __maxof2(double e1_, double e2_, double *p_, double *wk_, double *xv_);
double globalindepvar(double t, double *p_, double *wk_, double *xv_);
double __maxof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_);
double Vdot(double __V__, double __R__, double *p_, double *wk_, double *xv_);
double getbound(char *name, int which_bd, double *p_, double *wk_, double *xv_);
int heav(double x_, double *p_, double *wk_, double *xv_);
double initcond(char *varname, double *p_, double *wk_, double *xv_);
double __rhs_if(int cond_, double e1_, double e2_, double *p_, double *wk_, double *xv_);
void jacobian(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_);
double __minof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_);
double __minof2(double e1_, double e2_, double *p_, double *wk_, double *xv_);
double Rdot(double __V__, double __R__, double *p_, double *wk_, double *xv_);
double __minof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_);

int N_EVENTS = 0;
void assignEvents(EvFunType *events){
 
}

void auxvars(unsigned, unsigned, double, double*, double*, double*, unsigned, double*, unsigned, double*);
void jacobian(unsigned, unsigned, double, double*, double*, double**, unsigned, double*, unsigned, double*);
void jacobianParam(unsigned, unsigned, double, double*, double*, double**, unsigned, double*, unsigned, double*);
int N_AUXVARS = 0;


int N_EXTINPUTS = 0;


void vfieldfunc(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){

f_[0] = Rdot(V,R, p_, wk_, xv_);
f_[1] = Vdot(V,R, p_, wk_, xv_);

}




int getindex(char *name, double *p_, double *wk_, double *xv_) {

  if (strcmp(name, "R")==0)
	return 0;
  else if (strcmp(name, "V")==0)
	return 1;
  else if (strcmp(name, "a")==0)
	return 2;
  else if (strcmp(name, "b")==0)
	return 3;
  else if (strcmp(name, "c")==0)
	return 4;
  else {
	fprintf(stderr, "Invalid name %s for getindex call\n", name);
	return 0.0/0.0;
	}
}


double __maxof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_) {
double temp_;
if (e1_ > e2_) {temp_ = e1_;} else {temp_ = e2_;};
if (e3_ > temp_) {return e3_;} else {return temp_;};
}


double __maxof2(double e1_, double e2_, double *p_, double *wk_, double *xv_) {
if (e1_ > e2_) {return e1_;} else {return e2_;};
}


double globalindepvar(double t, double *p_, double *wk_, double *xv_) {
  return globalt0+t;
}


double __maxof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_) {
double temp_;
if (e1_ > e2_) {temp_ = e1_;} else {temp_ = e2_;};
if (e3_ > temp_) {temp_ = e3_;};
if (e4_ > temp_) {return e4_;} else {return temp_;};
}


double Vdot(double __V__, double __R__, double *p_, double *wk_, double *xv_) {


return c*(__V__-pow(__V__,3.)/3.+__R__);

}


double getbound(char *name, int which_bd, double *p_, double *wk_, double *xv_) {
  return gBds[which_bd][getindex(name, p_, wk_, xv_)];
}


int heav(double x_, double *p_, double *wk_, double *xv_) {
  if (x_>0.0) {return 1;} else {return 0;}
}


double initcond(char *varname, double *p_, double *wk_, double *xv_) {

  if (strcmp(varname, "R")==0)
	return gICs[0];
  else if (strcmp(varname, "V")==0)
	return gICs[1];
  else {
	fprintf(stderr, "Invalid variable name %s for initcond call\n", varname);
	return 0.0/0.0;
	}
}


double __rhs_if(int cond_, double e1_, double e2_, double *p_, double *wk_, double *xv_) {
  if (cond_) {return e1_;} else {return e2_;};
}


void jacobian(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {


f_[0][0] = c*(V+pow(V,2.));
f_[0][1] = -1/c;
f_[1][0] = c;
f_[1][1] = -b/c;

 ;

}


double __minof3(double e1_, double e2_, double e3_, double *p_, double *wk_, double *xv_) {
double temp_;
if (e1_ < e2_) {temp_ = e1_;} else {temp_ = e2_;};
if (e3_ < temp_) {return e3_;} else {return temp_;};
}


double __minof2(double e1_, double e2_, double *p_, double *wk_, double *xv_) {
if (e1_ < e2_) {return e1_;} else {return e2_;};
}


double Rdot(double __V__, double __R__, double *p_, double *wk_, double *xv_) {


return -(__V__+a-b*__R__)/c ;

}


double __minof4(double e1_, double e2_, double e3_, double e4_, double *p_, double *wk_, double *xv_) {
double temp_;
if (e1_ < e2_) {temp_ = e1_;} else {temp_ = e2_;};
if (e3_ < temp_) {temp_ = e3_;};
if (e4_ < temp_) {return e4_;} else {return temp_;};
}

void auxvars(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double *f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_){


}


void massMatrix(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}

void jacobianParam(unsigned n_, unsigned np_, double t, double *Y_, double *p_, double **f_, unsigned wkn_, double *wk_, unsigned xvn_, double *xv_) {
}
