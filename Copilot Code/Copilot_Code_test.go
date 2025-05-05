package main

import (
	"encoding/csv"
	"os"
	"strconv"
	"testing"
)

func BenchmarkLinearRegression(b *testing.B) {
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

		var data []Record
		for i, row := range records {
			if i == 0 {
				continue
			}
			x, _ := strconv.ParseFloat(row[1], 64)
			y, _ := strconv.ParseFloat(row[2], 64)
			rec := Record{
				Dataset: row[0],
				X:       x,
				Y:       y,
			}
			data = append(data, rec)
		}

		groups := groupByDataset(data)
		for _, records := range groups {
			_, _, err := simpleLinearRegression(records)
			if err != nil {
				b.Fatal(err)
			}
		}
	}
}
