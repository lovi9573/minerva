/*
 * FixedPoint.h
 *
 *  Created on: Jul 28, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_COMMON_FIXEDPOINT_H_
#define MINERVA_COMMON_FIXEDPOINT_H_


#if defined(FIXED_POINT) || defined(HAS_FPGA)

template<typename MULTYPE, typename TYPE, int FRACW>
class FixedPoint{
public:
	TYPE value;

	//static const MULTYPE MAXVAL = ~(((MULTYPE)~0) << ((sizeof(TYPE)*8)-1));
	//static const MULTYPE MINVAL = -(MULTYPE)(((TYPE)1) << ((sizeof(TYPE)*8)-1));
	static const MULTYPE MAXVAL =  (1 << ((sizeof(TYPE)*8) + FRACW-1))-1;
	static const MULTYPE MINVAL = -(1 << ((sizeof(TYPE)*8) + FRACW-1));
	static const MULTYPE MINPREC =     1 << FRACW;
	static const TYPE FRACTION_MASK = (1 << FRACW) - 1;


	/* Constructors */
	FixedPoint():value(0){}
	FixedPoint(int v):value((TYPE)(v<<FRACW)){}
	FixedPoint(double v):value(0){
		MULTYPE multiplier = ((MULTYPE)1)<< (2*FRACW);
		value = (TYPE)(round(v*multiplier) >> FRACW);
		//printf("%f (x%d)=> %d\n",v,multiplier,value);
	}
	FixedPoint(TYPE v):value(v){}
	//FixedPoint(MULTYPE v):value((TYPE)v){}


	/* Same type Operations */
	FixedPoint operator*(const FixedPoint& rhs){
		return FixedPoint(mul(this->value, rhs.value));
	}

	FixedPoint operator/(const FixedPoint& rhs){
		return FixedPoint(div(this->value, rhs.value));
	}

	FixedPoint operator+(const FixedPoint& rhs){
		return FixedPoint(add(this->value, rhs.value));
	}

	FixedPoint operator-(const FixedPoint& rhs){
		return FixedPoint(sub(this->value, rhs.value));
	}

	FixedPoint& operator*=(const FixedPoint& rhs){
		this->value = mul(this->value, rhs.value);
		return *this;
	}
	FixedPoint& operator/=(const FixedPoint& rhs){
		this->value = div(this->value, rhs.value);
		return *this;
	}
	FixedPoint& operator+=(const FixedPoint& rhs){
		this->value = add(this->value,rhs.value);
		return *this;
	}
	FixedPoint& operator-=(const FixedPoint& rhs){
		this->value = sub(this->value, rhs.value);
		return *this;
	}

	FixedPoint& operator=(const FixedPoint& other){
		this->value = other.value;
		return *this;
	}

	bool operator!(){
		return !value;
	}

	float to_float(){
		return ((float)this->value)/(1<<FRACW);
	}

	operator float(){
		return this->to_float();
	}

/*	operator double(){
		return (double)this->to_float();
	}*/

	const char* str() const{
		return (std::to_string(this->value >> FRACW) + "." + std::to_string(this->value & ((~(uint16_t)0) >> (sizeof(TYPE)*8 - FRACW)))).c_str();

	}

private:

	 inline MULTYPE round(MULTYPE val){
		MULTYPE adjustment =  (MULTYPE)rand() & (MULTYPE)FRACTION_MASK;
		//printf("Value: %d , Adjustment: %d\n",val,adjustment);
		val = val + adjustment;
		if(val > MAXVAL){
			printf("MAXVAL (%f) exceeded. truncating\n",((float)(MAXVAL >> FRACW))/(1<<FRACW));
			return MAXVAL ;
		}
		if(val < MINVAL){
			printf("MINVAL (%f) exceeded. truncating\n",((float)(MINVAL >> FRACW))/(1<<FRACW));
			return MINVAL;
		}
		TYPE out = (TYPE)(val >> FRACW);
		if(!out){
			printf("Underflow. (< %f)  Setting to minimum precision\n",(1.0f)/(1<<FRACW));
			return (MULTYPE)(1 << FRACW);
		}
		return val;
	 }

	 inline TYPE add(TYPE a, TYPE b){
			MULTYPE v = ((MULTYPE)a << FRACW)  + ((MULTYPE)b << FRACW) ;
			return (TYPE)(round(v) >> FRACW);
	 }

	 inline TYPE sub(TYPE a, TYPE b){
			MULTYPE v = ((MULTYPE)a << FRACW)  - ((MULTYPE)b << FRACW) ;
			return (TYPE)(round(v) >> FRACW);
	 }

	 inline TYPE mul(TYPE a, TYPE b){
			MULTYPE v = (MULTYPE)a * (MULTYPE)b;
			return (TYPE)(round(v) >> FRACW);
	 }

	 inline TYPE div(TYPE a, TYPE b){
			MULTYPE v = (((MULTYPE)a << FRACW) / (MULTYPE)b) << FRACW;
			return (TYPE)(round(v) >> FRACW);
	 }


	/* Bridge Operations */
	friend FixedPoint operator+(const FixedPoint& lhs, double rhs){
		return lhs + FixedPoint(rhs);
	}
	friend FixedPoint operator-(const FixedPoint& lhs, double rhs){
		return lhs - FixedPoint(rhs);
	}
	friend FixedPoint operator*(const FixedPoint& lhs, double rhs){
		return lhs * FixedPoint(rhs);
	}
	friend FixedPoint operator/(const FixedPoint& lhs, double rhs){
		return lhs / FixedPoint(rhs);
	}

	friend std::ostream& operator<<(std::ostream& os, const FixedPoint& obj){
		os << obj.value;
		return os;
	}

};

#endif /* fixed point */

#endif /* MINERVA_COMMON_FIXEDPOINT_H_ */
