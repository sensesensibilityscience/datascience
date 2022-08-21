/*
TODO
# Rearrange the circles into left-right halves for pos/neg like a histogram
*/

require.config({ 
    paths: { 
        d3src: 'https://d3js.org',
        slider: 'https://unpkg.com/d3-simple-slider@1.10.4/dist/d3-simple-slider.min'
    },
    map: {
        '*': {
            'd3': 'd3src/d3.v7.min',
            'd3-timer': 'd3src/d3-timer.v3.min',
            'd3-interpolate': 'd3src/d3-interpolate.v3.min',
            'd3-color': 'd3src/d3-color.v3.min',
            'd3-time': 'd3src/d3-time.v3.min',
            'd3-format': 'd3src/d3-format.v3.min',
            'd3-time-format': 'd3src/d3-time-format.v4.min',
            'd3-array': 'd3src/d3-array.v3.min',
            'd3-axis': 'd3src/d3-axis.v3.min',
            'd3-dispatch': 'd3src/d3-dispatch.v3.min',
            'd3-drag': 'd3src/d3-drag.v3.min',
            'd3-ease': 'd3src/d3-ease.v3.min',
            'd3-scale': 'd3src/d3-scale.v4.min',
            'd3-selection': 'd3src/d3-selection.v3.min',
            'd3-transition': 'd3src/d3-transition.v3.min'
        }
    }
})

var prior = 0.5
var true_pos = 0.5
var true_neg = 0.5

function drawSlider1(d3, slider, svg) {
    let g = svg.append('g').attr('transform', 'translate(39, 20)')
    g.call(
        slider.sliderBottom()
            .min(0).max(1)
            .width(200)
            .tickFormat(d3.format('.1%'))
            .tickValues([0, 0.25, 0.5, 0.75, 1])
            .default(.5)
            .on('onchange', val => {
                prior = val
                drawCircles(d3.select('#circles'))
            })
    )
}

function drawSlider2(d3, slider, svg) {
    let g = svg.append('g').attr('transform', 'translate(39, 20)')
    g.call(
        slider.sliderBottom()
            .min(0).max(1)
            .width(200)
            .tickFormat(d3.format('.1%'))
            .tickValues([0, 0.25, 0.5, 0.75, 1])
            .default(.5)
            .on('onchange', val => {
                true_pos = val
                drawCircles(d3.select('#circles'))
            })
    )
}

function drawSlider3(d3, slider, svg) {
    let g = svg.append('g').attr('transform', 'translate(39, 20)')
    g.call(
        slider.sliderBottom()
            .min(0).max(1)
            .width(200)
            .tickFormat(d3.format('.1%'))
            .tickValues([0, 0.25, 0.5, 0.75, 1])
            .default(.5)
            .on('onchange', val => {
                true_neg = val
                drawCircles(d3.select('#circles'))
            })
    )
}

function drawCircles(g) {
    let total = 1000
    let col_len = 25
    let d = 17
    // https://samanthaz.me/writing/finding-the-right-color-palettes-for-data-visualizations
    let pos_col = '#fadc64' // light yellow
    let neg_col = '#9fd5ce' // light blue
    let pos_stroke = '#e35d2e' // orange-red
    let neg_stroke = '#182574' // dark blue
    let data = []
    for (let i = 0; i < total; i++) {
        let x
        let y
        let r = 6
        let pos_neg_split = Math.round(total * prior)
        let fill = (i < pos_neg_split) ? pos_col : neg_col
        let stroke
        if (i < pos_neg_split) {
            x = ((i / col_len) >> 0) * d + 20
            y = (i % col_len) * d + 20
            stroke = (i < Math.round(total * prior * true_pos)) ? pos_stroke : neg_stroke
        } else {
            let i2 = i - pos_neg_split
            x = 830 - ((i2 / col_len) >> 0) * d
            y = (i2 % col_len * d + 20)
            stroke = (i < pos_neg_split + Math.round(total * (1-prior) * true_neg)) ? neg_stroke : pos_stroke
        }
        data[i] = [x, y, r, fill, stroke]
    }
    g.selectAll('circle')
        .data(data)
        .join('circle')
        .attr('cx', function(d, i) {
            return d[0]
        })
        .attr('cy', function(d, i) {
            return d[1]
        })
        .attr('r', function(d, i) {
            return d[2]
        })
        .attr('fill', function(d, i) {
            return d[3]
        })
        .attr('stroke', function(d, i) {
            return d[4]
        })
        .attr('stroke-width', 2.2)
}

require.undef('viz')
define('viz', ['d3', 'slider'], function(d3, slider) {
    function draw(container) {
        d3.select(container).append('div').attr('id', 'sliders')
        d3.select('#sliders').append('div').attr('id', 'slider1')
        d3.select('#sliders').append('div').attr('id', 'slider2')
        d3.select('#sliders').append('div').attr('id', 'slider3')
        d3.select('#slider1').append('div').attr('class', 'slider_label').text('Prior probability')
        d3.select('#slider2').append('div').attr('class', 'slider_label').text('True positive rate')
        d3.select('#slider3').append('div').attr('class', 'slider_label').text('True negative rate')
        let svg_slider1 = d3.select('#slider1').append('svg').attr('width', '280px').attr('height', '70px')
        drawSlider1(d3, slider, svg_slider1)
        let svg_slider2 = d3.select('#slider2').append('svg').attr('width', '280px').attr('height', '70px')
        drawSlider2(d3, slider, svg_slider2)
        let svg_slider3 = d3.select('#slider3').append('svg').attr('width', '280px').attr('height', '70px')
        drawSlider3(d3, slider, svg_slider3)
        d3.select(container).append('div').attr('id', 'graphic')
        d3.select('#graphic').append('svg').attr('width', '850px').attr('height', '600px')
        let g_circles = d3.select('#graphic svg').append('g').attr('id', 'circles')
        drawCircles(g_circles)
    }
    return draw;
});

element.append('Loaded 🎉')