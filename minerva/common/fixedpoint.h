/*
 * FixedPoint.h
 *
 *  Created on: Jul 28, 2015
 *      Author: jlovitt
 */

#ifndef MINERVA_COMMON_FIXEDPOINT_H_
#define MINERVA_COMMON_FIXEDPOINT_H_


#if defined(FIXED_POINT) || defined(HAS_FPGA)

#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <cmath>


template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
class FixedPoint{
public:
	TYPE value;

	//static const MULTYPE MAXVAL = ~(((MULTYPE)~0) << ((sizeof(TYPE)*8)-1));
	//static const MULTYPE MINVAL = -(MULTYPE)(((TYPE)1) << ((sizeof(TYPE)*8)-1));
	static const MULTYPE MAXVAL =  ((MULTYPE)1 << (WORDLEN ))-1;
	static const MULTYPE MINVAL = -((MULTYPE)1 << (WORDLEN ));
	//static const MULTYPE MINPREC =     (MULTYPE)1 << FRACW;
	static const TYPE FRACTION_MASK = ((TYPE)1 << FRACW) - 1;
	static const TYPE SIGN_MASK = MINVAL;

	static uint64_t n_MaxTruncs;
	static uint64_t n_MinTruncs;
	static uint64_t n_UnderFlows;


	/* Constructors */
	FixedPoint():value(0){}
	//FixedPoint(int v):value((TYPE)(v<<FRACW)){}
	inline FixedPoint(double v):value(0){
		MULTYPE multiplier = ((MULTYPE)1)<< (2*FRACW);
		value = (TYPE)trunc(round(v*multiplier) >> FRACW);
		//printf("%f (x%d)=> %d\n",v,multiplier,value);
	}
	explicit FixedPoint(TYPE v):value(v){}
	//FixedPoint(MULTYPE v):value((TYPE)v){}


	/* Same type Operations */
	inline FixedPoint operator*(const FixedPoint& rhs){
		return FixedPoint((TYPE)mul(this->value, rhs.value));
	}

	inline FixedPoint operator/(const FixedPoint& rhs){
		return FixedPoint(div(this->value, rhs.value));
	}

	inline FixedPoint operator+(const FixedPoint rhs){
		return FixedPoint((TYPE)add(this->value, rhs.value));
	}

	inline FixedPoint operator-(const FixedPoint& rhs){
		return FixedPoint(add(this->value, -rhs.value));
	}

	inline FixedPoint& operator*=(const FixedPoint& rhs){
		this->value = mul(this->value, rhs.value);
		return *this;
	}
	inline FixedPoint& operator/=(const FixedPoint& rhs){
		this->value = div(this->value, rhs.value);
		return *this;
	}
	inline FixedPoint& operator+=(const FixedPoint& rhs){
		this->value = add(this->value,rhs.value);
		return *this;
	}
	inline FixedPoint& operator-=(const FixedPoint& rhs){
		this->value = add(this->value, -rhs.value);
		return *this;
	}

	inline FixedPoint operator-(){
		return FixedPoint((TYPE)(-(this->value)));
	}

	inline FixedPoint& operator=(const FixedPoint& other){
		this->value = other.value;
		return *this;
	}

	inline bool operator==(const FixedPoint& other){
		return (this->value == other.value);
	}

	inline bool operator>(const FixedPoint& other){
		return (this->value > other.value);
	}

	inline bool operator<(const FixedPoint& other){
		return (this->value < other.value);
	}

	bool operator!(){
		return !value;
	}

	inline float to_float(){
		return ((float)this->value)/(1<<FRACW);
	}

	explicit operator float(){
		return this->to_float();
	}

	explicit operator double(){
		return (double)this->to_float();
	}

	static float getEpsilon(){
		return 1.0f/((float)(1 << FRACW));
	}

	const char* str() {
		std::ostringstream out;
		out << (this->to_float());
		return out.str().c_str();
	}

	static void Report(){
		std::cout << "# Maximum Overflows: " << n_MaxTruncs << "\n" \
				  << "# Minimum Overflows: " << n_MinTruncs << "\n" \
				  << "# Underflows: " << n_UnderFlows << "\n";
	}

private:

	 inline MULTYPE round(MULTYPE val){
		val = val + ((MULTYPE)rand() & (MULTYPE)FRACTION_MASK);
		return val ;
	 }

	 inline TYPE trunc(MULTYPE val){
		if(val > MAXVAL){
			//printf("MAXVAL (%f) exceeded. truncating\n",((float)(MAXVAL >> FRACW))/(1<<FRACW));
			n_MaxTruncs++;
			return (TYPE)MAXVAL ;
		}
		if(val < MINVAL){
			//printf("MINVAL (%f) exceeded. truncating\n",((float)(MINVAL >> FRACW))/(1<<FRACW));
			n_MinTruncs++;
			return (TYPE)MINVAL;
		}
		if(val == 0){
			//printf("Underflow. (< %f)  Setting to minimum precision\n",(1.0f)/(1<<FRACW));
			n_UnderFlows++;
			//return (MULTYPE)(1 << FRACW);
		}
		return (TYPE)val;
	 }



	 inline TYPE add(TYPE a, TYPE b){
			MULTYPE v = ((MULTYPE)a)  + ((MULTYPE)b) ;
			return trunc(v);
	 }

	 /*
	 inline TYPE sub(TYPE a, TYPE b){
			MULTYPE v = ((MULTYPE)a << FRACW)  - ((MULTYPE)b << FRACW) ;
			return (TYPE)(round(v) >> FRACW);
	 }
	  */
	 inline TYPE mul(TYPE a, TYPE b){
			MULTYPE v = (MULTYPE)a * (MULTYPE)b;
			return (TYPE)trunc(round(v) >> FRACW);
	 }

	 inline TYPE div(TYPE a, TYPE b){
			MULTYPE v = (((MULTYPE)a << FRACW) / (MULTYPE)b) ;
			return (TYPE)trunc(v) ;
	 }


	/* Bridge Operations */
	friend inline FixedPoint operator+(double lhs, const FixedPoint rhs){
		return FixedPoint(lhs) + rhs;
	}
	/*

	friend inline FixedPoint operator-(const FixedPoint& lhs, double rhs){
		return lhs - FixedPoint(rhs);
	}
	*/
	friend inline FixedPoint operator*(int lhs, const FixedPoint& rhs){
		return FixedPoint((double)lhs) * rhs;
	}
	friend inline FixedPoint operator/(double lhs, const FixedPoint& rhs){
		return FixedPoint(lhs)/ rhs;
	}


	friend std::ostream& operator<<(std::ostream& os, const FixedPoint& obj){
		os << obj.value;
		return os;
	}

	friend bool atMax(FixedPoint v){
		return (v.value == (TYPE)MAXVAL);
	}

};

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
uint64_t FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW>::n_MaxTruncs = 0;
template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
uint64_t FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW>::n_MinTruncs = 0;
template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
uint64_t FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW>::n_UnderFlows = 0;

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> pow(FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> a, FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> b){
	return pow((double)a, (double)b);
}

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> expf(FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> a){
	return FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW>(exp((double)a));
}

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> exp(FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> a){
	return exp((double)a);
}

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> log(FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> a){
	return log((double)a);
}

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> tanhf(FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> a){
	return tanh((double)a);
}

template<typename MULTYPE, typename TYPE, int WORDLEN, int FRACW>
FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW> abs(FixedPoint<MULTYPE,TYPE,WORDLEN,FRACW>& a){
	return std::abs((double)a);
}



#endif /* fixed point */

#endif /* MINERVA_COMMON_FIXEDPOINT_H_ */
