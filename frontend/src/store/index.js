import { configureStore } from '@reduxjs/toolkit';
import sentimentReducer from './sentimentSlice';

export const store = configureStore({
  reducer: {
    sentiment: sentimentReducer,
  },
});
