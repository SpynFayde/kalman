package kalman

import (
	mat "github.com/mrfyo/matrix"
)

type (
	Shape  = mat.Shape
	Matrix = mat.Matrix
)

type Filter interface {
	Predict()
	Update(z Matrix) (X Matrix)
}

type FilterFun func(dt float64, v ...interface{}) Matrix

type LinearKalmanFilter struct {
	DimX   int
	DimZ   int
	Dt     float64
	F      Matrix
	H      Matrix
	X      Matrix
	P      Matrix
	R      Matrix
	Q      Matrix
	PriorX Matrix
	PriorP Matrix
}

func (kf *LinearKalmanFilter) Init(x, P, Q, R Matrix) {
	kf.X = x
	kf.P = P
	kf.Q = Q
	kf.R = R
}

func (kf *LinearKalmanFilter) Predict() {
	X := kf.X
	P := kf.P
	F := kf.F
	Q := kf.Q

	kf.PriorX = F.Dot(X)
	kf.PriorP = F.Dot(P).Dot(F.T()).Add(Q)
}

func (kf *LinearKalmanFilter) Update(z Matrix) (X Matrix) {
	H := kf.H
	R := kf.R
	priorX := kf.PriorX
	priorP := kf.PriorP

	HT := H.T()
	S := H.Dot(priorP).Dot(HT).Add(R)
	K := priorP.Dot(HT).Dot(mat.Inv(S))
	y := z.Sub(H.Dot(priorX))

	kf.X = priorX.Add(K.Dot(y))
	kf.P = priorP.Sub(K.Dot(H).Dot(priorP))

	return kf.X
}

func NewKalmanFilter(dimX int, dimZ int, dt float64, F Matrix, H Matrix) *LinearKalmanFilter {

	shapeX := Shape{Row: dimX, Col: dimX}
	shapeZ := Shape{Row: dimZ, Col: dimZ}

	return &LinearKalmanFilter{
		DimX:   dimX,
		DimZ:   dimZ,
		Dt:     dt,
		F:      F,
		H:      H,
		X:      mat.Zeros(shapeX),
		P:      mat.Eye(dimX),
		R:      mat.Zeros(shapeZ),
		Q:      mat.Zeros(shapeX),
		PriorX: mat.Zeros(shapeX),
		PriorP: mat.Eye(dimX),
	}
}
