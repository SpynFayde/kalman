package kalman

import (
	"math"

	mat "github.com/mrfyo/matrix"
)

type CubatureKalmanFilter struct {
	DimX   int       // n
	DimZ   int       // m
	Dt     float64   // collect duration
	Fx     FilterFun // (n, 1)
	Hx     FilterFun // (m, 1)
	X      Matrix    // (n, 1)
	P      Matrix    // (n, n)
	R      Matrix    // (m, m)
	Q      Matrix    // (n, n)
	PriorX Matrix    // (n, 1)
	PriorP Matrix    // (n, n)
}

func (kf *CubatureKalmanFilter) Init(x, P, Q, R Matrix) {
	kf.X = x
	kf.P = P
	kf.Q = Q
	kf.R = R
}

// computeSigmaPoints 计算样本点
func (kf *CubatureKalmanFilter) computeSigmaPoints(x Matrix, P Matrix) (sigmas Matrix) {
	n := kf.DimX

	sigmas = mat.Zeros(Shape{Row: n, Col: 2 * n})

	L, _ := mat.Cholesky(P)

	S := L.ScaleMul(math.Sqrt(float64(n)))

	for k := 0; k < n; k++ {
		sigmas.SetCol(k, x.Add(S.GetCol(k)))
		sigmas.SetCol(k+n, x.Sub(S.GetCol(k)))
	}

	return sigmas
}

func (kf *CubatureKalmanFilter) Predict() {
	n := kf.DimX
	x := kf.X
	P := kf.P
	Q := kf.Q

	sigmas := kf.computeSigmaPoints(x, P)
	priorX := mat.Zeros(x.Shape)
	priorP := mat.Zeros(P.Shape)

	for j := 0; j < sigmas.Col; j++ {
		xk := kf.Fx(kf.Dt, sigmas.GetCol(j))
		mat.MatrixAdd(priorX, xk)
		mat.MatrixAdd(priorP, xk.Dot(xk.T()))
	}

	wn := 1.0 / float64(2*n)

	priorX = priorX.ScaleMul(wn)
	priorP = priorP.ScaleMul(wn).Sub(priorX.Dot(priorX.T())).Add(Q)

	kf.PriorX = priorX
	kf.PriorP = priorP
}

func (kf *CubatureKalmanFilter) Update(z Matrix) Matrix {
	n := kf.DimX
	m := kf.DimZ
	R := kf.R
	priorX := kf.PriorX
	priorP := kf.PriorP
	sigmas := kf.computeSigmaPoints(priorX, priorP) // (n, 2*n)

	Pzz := mat.Zeros(R.Shape)                  // (m, m)
	Pxz := mat.Zeros(Shape{Row: n, Col: m})    // (n, m)
	priorZ := mat.Zeros(Shape{Row: m, Col: 1}) // (m, 1)

	for k := 0; k < sigmas.Col; k++ {
		xk := sigmas.GetCol(k) // (n, 1)
		zk := kf.Hx(kf.Dt, xk) // (m, 1)
		zkT := zk.T()          // (1, m)
		mat.MatrixAdd(priorZ, zk)
		mat.MatrixAdd(Pzz, zk.Dot(zkT))
		mat.MatrixAdd(Pxz, xk.Dot(zkT))
	}
	wn := 1.0 / float64(2*n)

	priorZ = priorZ.ScaleMul(wn)

	priorZT := priorZ.T()
	Pzz = Pzz.ScaleMul(wn).Sub(priorZ.Dot(priorZT)).Add(R)
	Pxz = Pxz.ScaleMul(wn).Sub(priorX.Dot(priorZT))

	K := Pxz.Dot(mat.Inv(Pzz))             // (n, m)
	X := priorX.Add(K.Dot(z.Sub(priorZ)))  // (n, 1)
	P := priorP.Sub(K.Dot(Pzz).Dot(K.T())) // (n, n)

	kf.X = X
	kf.P = P
	return X
}

func NewCubatureKalmanFilter(dimX int, dimZ int, dt float64, Fx FilterFun, Hx FilterFun) *CubatureKalmanFilter {
	shapeX := Shape{Row: dimX, Col: dimX}
	shapeZ := Shape{Row: dimZ, Col: dimZ}

	return &CubatureKalmanFilter{
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