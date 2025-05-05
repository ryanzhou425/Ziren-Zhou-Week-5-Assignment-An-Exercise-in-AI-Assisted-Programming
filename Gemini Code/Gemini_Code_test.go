package main

import (
	"encoding/csv"
	"os"
	"strconv"
	"testing"
)

// Benchmark for Gemini-generated code using montanaflynn/stats
func BenchmarkGeminiLinearRegression(b *testing.B) {
	for i := 0; i < b.N; i++ {
		file, err := os.Open("anscombe.csv")
		if err != nil {
			b.Fatal(err)
		}
		reader := csv.NewReader(file)
		records, err := reader.ReadAll()
		if err != nil {
			b.Fatal(err)
		}
		file.Close()

		var data []DataPoint
		for i, row := range records {
			if i == 0 {
				continue
			}
			x, _ := strconv.ParseFloat(row[1], 64)
			y, _ := strconv.ParseFloat(row[2], 64)
			rec := DataPoint{
				Dataset: row[0],
				X:       x,
				Y:       y,
			}
			data = append(data, rec)
		}

		groups := groupDataByDataset(data)
		for _, records := range groups {
			_, _, err := calculateLinearRegression(records)
			if err != nil {
				b.Fatal(err)
			}
		}
	}
}
