import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

const API_URL = 'http://localhost:8000/api/predict';

export const analyzeSentiment = createAsyncThunk(
  'sentiment/analyze',
  async (text, { rejectWithValue }) => {
    try {
      const response = await axios.post(API_URL, { text });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || 'Something went wrong');
    }
  }
);

export const fetchHistory = createAsyncThunk(
  'sentiment/fetchHistory',
  async (_, { rejectWithValue }) => {
    try {
      const response = await axios.get('http://localhost:8000/api/history');
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch history');
    }
  }
);

export const analyzeUrl = createAsyncThunk(
  'sentiment/analyzeUrl',
  async (url, { rejectWithValue }) => {
    try {
      const response = await axios.post('http://localhost:8000/api/analyze-url', { url });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to analyze URL');
    }
  }
);

export const analyzeCompare = createAsyncThunk(
  'sentiment/analyzeCompare',
  async (urls, { rejectWithValue }) => {
    try {
      const response = await axios.post('http://localhost:8000/api/compare', { urls });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to compare URLs');
    }
  }
);

const sentimentSlice = createSlice({
  name: 'sentiment',
  initialState: {
    history: [],
    currentResult: null,
    urlResults: null,
    comparisonResults: null,
    status: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
    error: null,
  },
  reducers: {
    clearCurrentResult: (state) => {
      state.currentResult = null;
    },
    clearHistory: (state) => {
      state.history = [];
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(analyzeSentiment.pending, (state) => {
        state.status = 'loading';
        state.error = null;
      })
      .addCase(analyzeSentiment.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.currentResult = action.payload;
        state.history.unshift({
            ...action.payload,
            timestamp: new Date().toISOString()
        });
      })
      .addCase(analyzeSentiment.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      })
      .addCase(fetchHistory.fulfilled, (state, action) => {
        state.history = action.payload;
      })
      .addCase(analyzeUrl.pending, (state) => {
        state.status = 'loading';
        state.error = null;
        state.urlResults = null;
      })
      .addCase(analyzeUrl.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.urlResults = action.payload.results;
        // Optionally add all results to history
        state.history = [...action.payload.results, ...state.history];
      })
      .addCase(analyzeUrl.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      })
      .addCase(analyzeCompare.pending, (state) => {
        state.status = 'loading';
        state.error = null;
        state.comparisonResults = null;
      })
      .addCase(analyzeCompare.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.comparisonResults = action.payload.comparison;
      })
      .addCase(analyzeCompare.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      });
  },
});

export const { clearCurrentResult, clearHistory } = sentimentSlice.actions;
export default sentimentSlice.reducer;
