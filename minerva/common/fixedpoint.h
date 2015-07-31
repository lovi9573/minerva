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

	static const MULTYPE MAXVAL = ~(((MULTYPE)~0) << ((sizeof(TYPE)*8)-1));
	static const MULTYPE MINVAL = -(MULTYPE)(((TYPE)1) << ((sizeof(TYPE)*8)-1));

	FixedPoint():value(0){}
	FixedPoint(int v):value((TYPE)(v<<FRACW)){}
	FixedPoint(double v):value(0){
		value = v*(1<<FRACW);
		//printf("%f => %d\n",v,value);
	}
	FixedPoint(TYPE v):value(v){}
	//FixedPoint(MULTYPE v):value((TYPE)v){}

	FixedPoint operator*(const FixedPoint& rhs){
		printf("multiply %d * %d\n",this->value,rhs.value);
		return FixedPoint(((MULTYPE)this->value * (MULTYPE)rhs.value) >> FRACW);
	}

	FixedPoint operator/(const FixedPoint& rhs){
		return FixedPoint((((MULTYPE)this->value) << FRACW) / (MULTYPE)rhs.value);
	}

	FixedPoint operator+(const FixedPoint& rhs){
		return FixedPoint(add(this->value,rhs.value));
	}

	FixedPoint operator-(const FixedPoint& rhs){
		MULTYPE v = (MULTYPE)this->value - (MULTYPE)rhs.value;
		if(v > MAXVAL){
			return FixedPoint(MAXVAL);
		}
		if(v < MINVAL){
			return FixedPoint(MINVAL);
		}
		return FixedPoint(v);
	}

	FixedPoint operator*=(const FixedPoint& rhs){
		return FixedPoint(*this*rhs);
	}
	FixedPoint operator/=(const FixedPoint& rhs){
			return FixedPoint(*this/rhs);
	}
	FixedPoint& operator+=(const FixedPoint& rhs){
		//TYPE tmp = this->value;
		this->value = add(this->value,rhs.value);
		//printf("%d += %d = %d\n",tmp ,rhs.value,this->value);
		return *this;
	}
	FixedPoint operator-=(const FixedPoint& rhs){
			return FixedPoint(*this-rhs);
	}

	FixedPoint& operator=(const FixedPoint& other){
		this->value = other.value;
		return *this;
	}

	bool operator!(){
		return !value;
	}

	float to_float(){
		//printf("%d converted to float by / (1 << %d)\n",this->value,FRACW);
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
	 inline TYPE add(TYPE a, TYPE b){
			MULTYPE v = (MULTYPE)a + (MULTYPE)b;
			//printf("%d + %d = %d\n",a, b,v);
			//printf("%d < %d < %d\n",MINVAL,v,MAXVAL);
			if(v > MAXVAL){
				printf("MAXTRUNC\n");
				return (TYPE)MAXVAL;
			}
			if(v < MINVAL){
				printf("MINTRUNC\n");
				return (TYPE)MINVAL;
			}
			return (TYPE)v;
	 }

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
