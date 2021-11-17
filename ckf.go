package kalman

import (
	"math"

	mat "github.com/mrfyo/matrix"
)

type UnscentedKalmanFilter struct {
	DimX        int       // n
	DimZ        int       // m
	Dt          float64   // collect duration
	Fx          FilterFun // (n, 1)
	Hx          FilterFun // (m, 1)
	X           Matrix    // (n, 1)
	P           Matrix    // (n, n)
	R           Matrix    // (m, n)
	Q           Matrix    // (n, n)
	PriorX      Matrix    // (n, 1)
	PriorP      Matrix    // (n, n)
	Wm          Matrix    // (1, 2*n+1)
	Wc          Matrix    // (1, 2*n+1)
	SigmaXs     Matrix    // (n, 2*n+1)
	SigmaParams []float64 //(alpha, kappa, beta)
}

func (kf *UnscentedKalmanFilter) Init(x, P, Q, R Matrix) {
	kf.X = x
	kf.P = P
	kf.Q = Q
	kf.R = R
	kf.SigmaParams = []float64{0.1, 0, 2.0}
	kf.Wm, kf.Wc = kf.computeSigmaWeights(x, P)
}

func (kf *UnscentedKalmanFilter) computeSigmaWeights(x, P Matrix) (Wm, Wc Matrix) {
	n := kf.DimX
	alpha := kf.SigmaParams[0]
	kappa := kf.SigmaParams[1]
	beta := kf.SigmaParams[2]

	nf := float64(n)
	_lambda := (alpha*alpha)*(nf+kappa) - nf

	m := 2*n + 1
	c := 0.5 / (nf + _lambda)
	Wm = mat.Full(Shape{Row: 1, Col: m}, c)
	Wc = mat.Full(Shape{Row: 1, Col: m}, c)

	Wm.Set(0, 0, _lambda/(nf+_lambda))
	Wc.Set(0, 0, _lambda/(nf+_lambda)+(1-alpha*alpha+beta))

	return
}

func (kf *UnscentedKalmanFilter) computeSigmaPoints(x, P Matrix) (sigmas Matrix) {
	n := kf.DimX
	alpha := kf.SigmaParams[0]
	kappa := kf.SigmaParams[1]

	m := 2*n + 1
	sigmas = mat.Zeros(Shape{Row: n, Col: m}) // (n, 2*n+1)
	sigmas.SetCol(0, x)

	_lambda := (alpha*alpha)*(float64(n)+kappa) - float64(n)

	_, L := mat.Cholesky(P)
	s := math.Sqrt(_lambda + float64(n))
	S := L.ScaleMul(s)

	sigmas.SetCol(0, x)
	for k := 0; k < n; k++ {
		Sk := S.GetCol(k)
		sigmas.SetCol(k+1, x.Sub(Sk))
		sigmas.SetCol(k+n+1, x.Add(Sk))
	}

	return
}

func (kf *UnscentedKalmanFilter) Predict() {
	n := kf.DimX
	x := kf.X
	P := kf.P
	Q := kf.Q
	Wm := kf.Wm
	Wc := kf.Wc

	sigmas := kf.computeSigmaPoints(x, P) // (2*n+1, n)
	count := sigmas.Col

	sigmaXs := mat.Zeros(sigmas.Shape) // (2*n+1, n)
	for j := 0; j < count; j++ {
		Y := kf.Fx(kf.Dt, sigmas.GetCol(j))
		sigmaXs.SetCol(j, Y)
	}

	priorX := mat.Zeros(Shape{Row: n, Col: 1})
	for j := 0; j < count; j++ {
		for i := 0; i < sigmaXs.Row; i++ {
			v := sigmaXs.Get(i, j)*Wm.Get(0, j) + priorX.Get(i, 0)
			priorX.Set(i, 0, v)
		}
	}

	priorP := Q.Copy() // (n, n)
	for j := 0; j < count; j++ {
		diffX := sigmaXs.GetCol(j).Sub(priorX) // (n, 1)
		w := Wc.Get(0, j)
		mat.MatrixAdd(priorP, diffX.Dot(diffX.T()).ScaleMul(w))
	}

	kf.PriorX = priorX
	kf.PriorP = priorP
	kf.SigmaXs = sigmaXs

}

func (kf *UnscentedKalmanFilter) Update(z Matrix) Matrix {
	n, m, Wm, Wc := kf.DimX, kf.DimZ, kf.Wm, kf.Wc
	R := kf.R
	priorX := kf.PriorX   // (n, 1)
	priorP := kf.PriorP   // (n, n)
	sigmaXs := kf.SigmaXs // (n, 2*n+1)

	count := sigmaXs.Col
	sigmaZs := mat.Zeros(Shape{
		Row: m,
		Col: count,
	})
	for j := 0; j < count; j++ {
		sigmaZs.SetCol(j, kf.Hx(kf.Dt, sigmaXs.GetCol(j)))
	}

	priorZ := mat.Zeros(Shape{Row: m, Col: 1})
	for j := 0; j < count; j++ {
		w := Wm.Get(0, j)
		mat.MatrixAdd(priorZ, sigmaZs.GetCol(j).ScaleMul(w))
	}

	Pzz := R.Copy()                         // (m, m)
	Pxz := mat.Zeros(Shape{Row: n, Col: m}) // (n, m)

	for j := 0; j < count; j++ {
		w := Wc.Get(0, j)
		diffZ := sigmaZs.GetCol(j).Sub(priorZ) // (m, 1)
		mat.MatrixAdd(Pzz, diffZ.Dot(diffZ.T()).ScaleMul(w))
		diffX := sigmaXs.GetCol(j).Sub(priorX) // (n, 1)
		mat.MatrixAdd(Pxz, diffX.Dot(diffZ.T()).ScaleMul(w))
	}

	K := Pxz.Dot(mat.Inv(Pzz)) // (n, m)
	X := priorX.Add(K.Dot(z.Sub(priorZ)))
	P := priorP.Sub(K.Dot(Pzz).Dot(K.T()))

	kf.X = X
	kf.P = P
	return X
}

func NewUnscentedKalmanFilter(dimX int, dimZ int, dt float64, Fx FilterFun, Hx FilterFun) *UnscentedKalmanFilter {
	shapeX := Shape{Row: dimX, Col: dimX}
	shapeZ := Shape{Row: dimZ, Col: dimZ}

	return &UnscentedKalmanFilter{
		DimX:   dimX,
		DimZ:   dimZ,
		Dt:     dt,
		Fx:     Fx,
		Hx:     Hx,
		X:      mat.Zeros(shapeX),
		P:      mat.Eye(dimX),
		R:      mat.Zeros(shapeZ),
		Q:      mat.Zeros(shapeX),
		PriorX: mat.Zeros(shapeX),
		PriorP: mat.Eye(dimX),
	}
}
