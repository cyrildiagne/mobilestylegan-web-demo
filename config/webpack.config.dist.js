const path = require('path')
const CopyPlugin = require("copy-webpack-plugin")
const MiniCssExtractPlugin = require('mini-css-extract-plugin')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const HTMLInlineCSSWebpackPlugin =
  require('html-inline-css-webpack-plugin').default

module.exports = {
  entry: './src/index.ts',
  mode: 'production',
  output: {
    path: path.resolve(__dirname, '../dist'),
    filename: 'bundle.js',
    clean: true,
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "public",
          // prevents the index.html from being copied to the the public folder,
          // as it's going to be generated by webpack
          filter: async (filePath) => {
            return path.basename(filePath) !== "index.html"
          }
        }
      ]
    }),
    new MiniCssExtractPlugin({
      filename: '[name].css',
      chunkFilename: '[id].css',
    }),
    new HtmlWebpackPlugin({
      template: './public/index.html',
      inject: 'body',
      publicPath: './',
    }),
    new HTMLInlineCSSWebpackPlugin(),
  ],
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        // exclude: /node_modules/,
      },
      {
        test: /\.glsl$/,
        use: 'webpack-glsl-loader',
      },
    ],
  },
}