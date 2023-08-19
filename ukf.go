package kalman

import (
	"errors"
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
	R           Matrix    // (m, m)
	Q           Matrix    // (n, n)
	PriorX      Matrix    // (n, 1)
	PriorP      Matrix    // (n, n)
	Wm          Matrix    // (1, 2*n+1)
	Wc          Matrix    // (1, 2*n+1)
	SigmaXs     Matrix    // (n, 2*n+1)
	SigmaParams []float64 //(alpha, kappa, beta)
	S           Matrix    // System uncertainty
	Y           Matrix    // Residual
	Adaptations []int
	phi         []float64
}

func (kf *UnscentedKalmanFilter) Init(x, P, Q, R Matrix) {
	kf.X = x
	kf.P = P
	kf.Q = Q
	kf.R = R
	kf.SigmaParams = []float64{0.1, 0, 2.0}
	kf.Wm, kf.Wc = kf.computeSigmaWeights()
}

func (kf *UnscentedKalmanFilter) computeSigmaWeights() (Wm, Wc Matrix) {
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

	L, _ := mat.Cholesky(P)
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

	sigmas := kf.computeSigmaPoints(x, P) // (n, 2*n+1)
	c := sigmas.Col

	sigmaXs := mat.Zeros(sigmas.Shape) // (n, 2*n+1)
	for j := 0; j < c; j++ {
		Y := kf.Fx(kf.Dt, sigmas.GetCol(j))
		sigmaXs.SetCol(j, Y)
	}

	priorX := mat.Zeros(Shape{Row: n, Col: 1})
	for j := 0; j < c; j++ {
		for i := 0; i < sigmaXs.Row; i++ {
			v := sigmaXs.Get(i, j)*Wm.Get(0, j) + priorX.Get(i, 0)
			priorX.Set(i, 0, v)
		}
	}

	priorP := Q.Copy() // (n, n)
	for j := 0; j < c; j++ {
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
	Y := z.Sub(priorZ)
	X := priorX.Add(K.Dot(Y))
	P := priorP.Sub(K.Dot(Pzz).Dot(K.T()))

	kf.S = Pzz
	kf.Y = Y
	kf.X = X
	kf.P = P
	return X
}

// Adapt uses system uncertainty to adjust process noise by a scale factor
// as it determines a maneuver has been made based on a factor of standard
// deviation of the system uncertainty
func (kf *UnscentedKalmanFilter) Adapt(dt, stdScale, qScaleFactor float64) {
	y := kf.Y
	s := kf.S

	// System uncertainty has multiple dimensions of data -- adapt per dimension
	for i := 0; i < kf.DimZ; i++ {
		std := math.Sqrt(s.Get(i, i))

		if math.Abs(y.GetIndex(i)) > stdScale*std {
			kf.phi[i] += qScaleFactor
			noise, _ := QDiscreteWhiteNoise(kf.DimX, dt, kf.phi[i], 1, true)
			kf.Q = BlockDiag(noise, noise, noise)
			kf.Adaptations[i] += 1
		} else if kf.Adaptations[i] > 0 {
			kf.phi[i] -= qScaleFactor
			noise, _ := QDiscreteWhiteNoise(kf.DimX, dt, kf.phi[i], 1, true)
			kf.Q = BlockDiag(noise, noise, noise)
			kf.Adaptations[i] -= 1
		}
	}
}

func OrderByDerivative(q []float64, dim int, blockSize int) mat.Matrix {
	N := dim * blockSize
	D := mat.Zeros(mat.Shape{Row: N, Col: N})
	for i, x := range q {
		f := mat.Eye(blockSize).ScaleMul(x)

		ix, iy := (i/dim)*blockSize, (i%dim)*blockSize
		for fi := 0; fi < f.Row; fi++ {
			for fj := 0; fj < f.Col; fj++ {
				D.Set(ix+fi, iy+fj, f.Get(fi, fj))
			}
		}
	}
	return D
}

func QDiscreteWhiteNoise(dim int, dt float64, variance float64, blockSize int, orderByDim bool) (mat.Matrix, error) {
	if dim != 2 && dim != 3 && dim != 4 {
		return mat.Zeros(mat.Shape{Row: dim, Col: dim}), errors.New("dim must be between 2 and 4")
	}

	var Q []float64
	switch dim {
	case 2:
		Q = []float64{
			0.25 * math.Pow(dt, 4), 0.5 * math.Pow(dt, 3),
			0.5 * math.Pow(dt, 3), math.Pow(dt, 2),
		}
	case 3:
		Q = []float64{
			0.25 * math.Pow(dt, 4), 0.5 * math.Pow(dt, 3), 0.5 * math.Pow(dt, 2),
			0.5 * math.Pow(dt, 3), math.Pow(dt, 2), dt,
			0.5 * math.Pow(dt, 2), dt, 1,
		}
	default:
		Q = []float64{
			math.Pow(dt, 6) / 36, math.Pow(dt, 5) / 12, math.Pow(dt, 4) / 6, math.Pow(dt, 3) / 6,
			math.Pow(dt, 5) / 12, math.Pow(dt, 4) / 4, math.Pow(dt, 3) / 2, math.Pow(dt, 2) / 2,
			math.Pow(dt, 4) / 6, math.Pow(dt, 3) / 2, math.Pow(dt, 2), dt,
			math.Pow(dt, 3) / 6, math.Pow(dt, 2) / 2, dt, 1,
		}
	}

	if orderByDim {
		QMat := mat.NewMatrix(mat.Shape{Row: dim, Col: dim}, Q)
		QMat.ScaleMul(float64(blockSize))
		res := BlockDiag(QMat)
		res.ScaleMul(variance)
		return res, nil
	}

	res := OrderByDerivative(Q, dim, blockSize)
	res.ScaleMul(variance)
	return res, nil
}

func BlockDiag(mats ...mat.Matrix) mat.Matrix {
	w := 0
	h := 0
	for _, matrix := range mats {
		mw, mh := matrix.Shape.Row, matrix.Shape.Col
		w += mw
		h += mh
	}

	newMatrix := mat.Zeros(mat.Shape{Row: w, Col: h})
	w, h = 0, 0
	for _, matrix := range mats {
		mw, mh := matrix.Shape.Row, matrix.Shape.Col
		for i := 0; i < mw; i++ {
			for j := 0; j < mh; j++ {
				newMatrix.Set(w+i, h+j, matrix.Get(i, j))
			}
		}
		w += mw
		h += mh
	}

	return newMatrix
}

func NewUnscentedKalmanFilter(dimX int, dimZ int, dt float64, Fx FilterFun, Hx FilterFun) *UnscentedKalmanFilter {
	shapeX := Shape{Row: dimX, Col: dimX}
	shapeZ := Shape{Row: dimZ, Col: dimZ}
	phi := make([]float64, dimZ)
	for x := range phi {
		phi[x] = 0.02
	}

	return &UnscentedKalmanFilter{
		DimX:        dimX,
		DimZ:        dimZ,
		Dt:          dt,
		Fx:          Fx,
		Hx:          Hx,
		X:           mat.Zeros(shapeX),
		P:           mat.Eye(dimX),
		R:           mat.Zeros(shapeZ),
		Q:           mat.Zeros(shapeX),
		PriorX:      mat.Zeros(shapeX),
		PriorP:      mat.Eye(dimX),
		phi:         phi,
		Adaptations: make([]int, dimZ),
	}
}
