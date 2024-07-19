package utilities;
// mutable 2D vectors
public final class Vector2D {
public double x, y;

// constructor for zero vector
public Vector2D() {
	this.x = 0;
	this.y = 0;
}

// constructor for vector with given coordinates
public Vector2D(double x, double y) {
	this.x = x;
	this.y  = y;	
}

// constructor that copies the argument vector 
public Vector2D(Vector2D v) {
	this.x = v.x;
	this.y = v.y;
}

// set coordinates
public Vector2D set(double x, double y) {
	this.x = x;
	this.y  =y;
	return this;
}

// set coordinates based on argument vector 
public Vector2D set(Vector2D v) {
	this.x = v.x;
	this.y = v.y;
	return this;
}

// compare for equality (note Object type argument) 
public boolean equals(Object o) {
	Vector2D b = (Vector2D) o;
	if(this.x == b.x && this.y == b.y)return true;
	else return false;
	} 

// String for displaying vector as text 
public String toString() {
	StringBuffer sb = new StringBuffer("x: <"+this.x+"> , y: <"+this.y);
	return sb+">";
	
}
		
//  magnitude (= "length") of this vector 
public double mag() {
	return Math.hypot(this.x, this.y);
	
}

// angle between vector and horizontal axis in radians in range [-PI,PI] 
// can be calculated using Math.atan2 
public double angle() {
	return Math.atan2(this.y, this.x);
}

// angle between this vector and another vector in range [-PI,PI] 
public double angle(Vector2D other) {
	double dot = this.dot(other);
  double diff = other.angle() -  this.angle();
  if (diff > Math.PI){
      diff -= 2 * Math.PI;
  }else if (diff < -Math.PI){
      diff += 2 * Math.PI;
  }
  return diff;

}
public void EQUAL(Vector2D other){
	this.x = other.x;
	this.y = other.y;
}
// add argument vector 
public Vector2D add(Vector2D v) {
	this.x += v.x;
	this.y += v.y;
	
	return this;
}

// add values to coordinates 
public Vector2D add(double x, double y) {
	this.x += x;
	this.y += y;
	return this;
}

// weighted add - surprisingly useful
public Vector2D addScaled(Vector2D v, double fac) {
	this.x = this.x + v.x*fac;
	this.y = this.y + v.y*fac;
	return this;
}

// subtract argument vector 
public Vector2D subtract(Vector2D v) {
	this.x -= v.x;
	this.y -= v.y;
	return this;
}

// subtract values from coordinates 
public Vector2D subtract(double x, double y) {
	this.x -= x;
	this.y -= y;
	return this;
}

// multiply with factor 
public Vector2D mult(double fac) {
	this.x *= fac;
	this.y *= fac;
	return this;
	
	
}
public Vector2D SpeedCheck(){
	if(this.mag() >= 400){
		if(Math.abs(this.x) > Math.abs(this.y)){
			if(this.x > 0){
				this.x = 400;
			}
			if(this.x < 0){
			this.x = -400;
		}
		}
		if(Math.abs(this.x) < Math.abs(this.y)){
			if(this.y > 0){
				this.y =400;
			}
			if(this.y <0){
				this.y = -400;
			}
		}
}
	return this;
}
// rotate by angle given in radians 
public Vector2D rotate(double angle) {
	double cost = Math.cos(angle);
	double sint = Math.sin(angle);
	double nx = x*cost - y*sint;
	double ny = x*sint + y*cost;
	this.x = nx;
	this.y = ny;
	return this;
}
public Vector2D subrotate(double angle){
	double cost = Math.cos(angle);
	double sint = Math.sin(angle);
	double nx = x*cost - y*sint;
	double ny = x*sint + y*cost;
	Vector2D x = new Vector2D(nx,ny);
	return x;
}
public Vector2D boarder(double w, double h , double radius ){
	if(this.x+ radius  >= w) this.x = w-radius;
	if(this.x- radius <= 0) this.x = 0+radius;
	if (this.y+ radius >= h) this.y = h-radius;
	if (this.y-radius <= 0) this.y = 0+radius;
	return this;
}
// "dot product" ("scalar product") with argument vector 
public double dot(Vector2D v) {
	double a = (this.x * v.x )+ (this.y * v.y);
	return a;
}

// distance to argument vector 
public double dist(Vector2D v) {
	double a = this.x -v.x;
	double b = this.y - v.y;
	return Math.hypot(a, b);
}

// normalise vector so that magnitude becomes 1 
public Vector2D normalise() {
	double mag = this.mag();
	this.x /= mag;
	this.y /= mag;
	return this;
}

// wrap-around operation, assumes w> 0 and h>0
public Vector2D wrap(double w, double h) {
	
	x = (x+w)%w;
	y = (y+h)%h;
	return this;
}

// construct vector with given polar coordinates  
public static Vector2D polar(double angle, double mag) {
	return new Vector2D(mag*Math.cos(angle), mag*Math.sin(angle));
}
}
