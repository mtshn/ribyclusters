package ru.ac.phyche.ribyclusters;

public abstract class ColumnFeatures {
	public abstract float[] columnFeatures(int column);

	public float[][] columnFeatures(int[] columns) {
		float[][] result = new float[columns.length][];
		for (int i = 0; i < result.length; i++) {
			result[i] = columnFeatures(columns[i]);
		}
		return result;
	}

	public static class NonpolarColumnShortFeatures extends ColumnFeatures {
		@Override
		public float[] columnFeatures(int column) {
			float[] result = new float[3];
			if (column < 15) {
				result[1] = 1.0f;
			} else {
				result[2] = 1.0f;
			}
			if ((column == 20) || (column == 24)) {
				result[0] = 1;
				result[2] = 0;
			}
			return result;
		}
	}

	public static class NonpolarColumnLongFeatures extends ColumnFeatures {

		@Override
		public float[] columnFeatures(int column) {
			float[] result = new float[39];
			if (column < 15) {
				result[1] = 1.0f;
			} else {
				result[2] = 1.0f;
			}
			if ((column == 20) || (column == 24)) {
				result[0] = 1;
				result[2] = 0;
			}
			result[column + 3] = 1;
			return result;
		}

	}

	public static class PolarColumnLongFeatures extends ColumnFeatures {

		@Override
		public float[] columnFeatures(int column) {
			float[] result = new float[21];
			result[column - 15] = 1.0f;
			return result;
		}

	}

	public static class NoColumnFeatures extends ColumnFeatures {
		@Override
		public float[] columnFeatures(int column) {
			return new float[] {};
		}

	}

}
