package main

import (
	"fmt"
	"io/fs"
	"io/ioutil"
	"math"
	"math/rand"
	"strings"

	"github.com/SpynFayde/kalman"
	"github.com/mrfyo/matrix"
)

func main() {
	dimX := 4
	dimZ := 2
	dt := 1.0

	QStd := 0.001
	RStd := 0.4

	fx := func(dt float64, X matrix.Matrix) matrix.Matrix {
		F := matrix.Builder().Row().
			Link(1, dt, 0, 0).
			Link(0, 1, 0, 0).
			Link(0, 0, 1, dt).
			Link(0, 0, 0, 1).Build()
		return F.Dot(X)
	}

	hx := func(dt float64, X matrix.Matrix) matrix.Matrix {
		H := matrix.Builder().Row().
			Link(1, 0, 0, 0).
			Link(0, 0, 1, 0).Build()
		return H.Dot(X)
	}

	X := matrix.Builder().Col().Link(0, 1, 0, 0.5).Build()

	P := matrix.Diag([]float64{10, 4, 10, 4})

	Q := matrix.Builder().Row().
		Link(0.25*math.Pow(dt, 4), 0.5*math.Pow(dt, 3), 0, 0).
		Link(0.5*math.Pow(dt, 3), dt*dt, 0, 0).
		Link(0, 0, 0.25*math.Pow(dt, 4), 0.5*math.Pow(dt, 3)).
		Link(0, 0, 0.5*math.Pow(dt, 3), dt*dt).
		Build().
		ScaleMul(QStd)

	R := matrix.Eye(dimZ).ScaleMul(RStd * RStd)

	kf := kalman.NewCubatureKalmanFilter(dimX, dimZ, dt, fx, hx)
	kf.Init(X, P, Q, R)

	N := 20
	D := CreateTrack(N, X.GetIndex(0), X.GetIndex(1), X.GetIndex(2), X.GetIndex(3), dt, RStd, QStd)

	var result []string
	for i := 1; i < N; i++ {
		x := D.Get(i, 2)
		y := D.Get(i, 3)
		z := matrix.NewVector([]float64{x, y}, 1)

		kf.Predict()
		XZ := kf.Update(z)

		csv := fmt.Sprintf("%.5f,%.5f,%.5f,%.5f,%.5f,%.5f", D.Get(i, 0), D.Get(i, 1), x, y, XZ.GetIndex(0), XZ.GetIndex(2))
		result = append(result, csv)
	}

	csv := strings.Join(result, "\r\n")

	ioutil.WriteFile("./out.csv", []byte(csv), fs.ModePerm)

}

func CreateTrack(n int, x, vx, y, vy, dt, RStd, VStd float64) (Z matrix.Matrix) {
	Z = matrix.Zeros(matrix.Shape{Row: n, Col: 4})

	for i := 0; i < n; i++ {
		x += (vx + rand.NormFloat64()*VStd) * dt
		y += (vy + rand.NormFloat64()*VStd) * dt
		Z.Set(i, 0, x)
		Z.Set(i, 1, y)
		Z.Set(i, 2, x+RStd*rand.NormFloat64())
		Z.Set(i, 3, y+RStd*rand.NormFloat64())
	}
	return
}
