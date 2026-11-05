from typing import TypedDict, cast

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType

from mlops.validation import RANGES, validate

COLUMNS = ["Id", *RANGES]
SCHEMA = StructType(
    [StructField(name, DoubleType(), nullable=True) for name in COLUMNS]
)
VALID = {
    "OverallQual": 7.0,
    "GrLivArea": 1800.0,
    "GarageCars": 2.0,
    "TotalBsmtSF": 900.0,
    "YearBuilt": 2005.0,
    "SalePrice": 200_000.0,
    "YrSold": 2008.0,
    "MoSold": 6.0,
}


class HousingRow(TypedDict, total=False):
    Id: float
    OverallQual: float | None
    GrLivArea: float | None
    GarageCars: float | None
    TotalBsmtSF: float | None
    YearBuilt: float | None
    SalePrice: float | None
    YrSold: float | None
    MoSold: float | None


def make_rows(count: int = 20, **overrides: float | None) -> list[HousingRow]:
    return cast(
        list[HousingRow],
        [{**VALID, "Id": float(i), **overrides} for i in range(count)],
    )


def test_valid_data_passes(spark: SparkSession) -> None:
    report = validate(
        spark.createDataFrame(make_rows(), schema=SCHEMA),  # type: ignore[reportUnknownMemberType]
        min_rows=10,
    )
    assert report.passed, report.failures


def test_missing_column_fails(spark: SparkSession) -> None:
    df = spark.createDataFrame(  # type: ignore[reportUnknownMemberType]
        make_rows(), schema=SCHEMA
    ).drop("GrLivArea")
    report = validate(df, min_rows=10)
    assert not report.passed
    assert "GrLivArea" in report.failures[0]


def test_too_few_rows_fails(spark: SparkSession) -> None:
    report = validate(
        spark.createDataFrame(make_rows(5), schema=SCHEMA),  # type: ignore[reportUnknownMemberType]
        min_rows=10,
    )
    assert not report.passed


def test_out_of_range_value_fails(spark: SparkSession) -> None:
    report = validate(
        spark.createDataFrame(  # type: ignore[reportUnknownMemberType]
            make_rows(OverallQual=42.0), schema=SCHEMA
        ),
        min_rows=10,
    )
    assert any("OverallQual" in failure for failure in report.failures)


def test_too_many_missing_values_fails(spark: SparkSession) -> None:
    rows = make_rows()
    for row in rows[:5]:  # 25% missing, limit is 5%
        row["GrLivArea"] = None
    report = validate(
        spark.createDataFrame(rows, schema=SCHEMA),  # type: ignore[reportUnknownMemberType]
        min_rows=10,
    )
    assert any("GrLivArea" in failure for failure in report.failures)


def test_duplicate_ids_fail(spark: SparkSession) -> None:
    report = validate(
        spark.createDataFrame(make_rows(Id=1.0), schema=SCHEMA),  # type: ignore[reportUnknownMemberType]
        min_rows=10,
    )
    assert report.duplicate_ids == 19
    assert not report.passed
