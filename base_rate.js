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

var total = 1000
var prior = 0.5
var true_pos = 0.5
var true_neg = 0.5
var graphic_width = 850
const pos_col = '#fadc64' // light yellow
const neg_col = '#9fd5ce' // light blue
const pos_stroke = '#e35d2e' // orange-red
const neg_stroke = '#182574' // dark blue

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
                legendText(d3, d3.select('#legend'))
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
                legendText(d3, d3.select('#legend'))
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
                legendText(d3, d3.select('#legend'))
            })
    )
}

function drawCircles(g) {
    let col_len = 25
    let d = 17
    // https://samanthaz.me/writing/finding-the-right-color-palettes-for-data-visualizations
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
            x = graphic_width - 20 - ((i2 / col_len) >> 0) * d
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

function drawLegend(g) {
    let y = 500
    let r = 8
    let stroke_width = 2.2/6*8
    g.append('text')
        .attr('id', 'total_text')
        .attr('x', graphic_width/2)
        .attr('y', y-24)
        .attr('text-anchor', 'middle')
        .text('Out of ' + total + ', there are...')
    g.append('circle')
        .attr('cx', 50)
        .attr('cy', y)
        .attr('r', r)
        .attr('fill', pos_col)
        .attr('stroke', pos_stroke)
        .attr('stroke-width', stroke_width)
    g.append('circle')
        .attr('cx', 250)
        .attr('cy', y)
        .attr('r', r)
        .attr('fill', pos_col)
        .attr('stroke', neg_stroke)
        .attr('stroke-width', stroke_width)
    g.append('circle')
        .attr('cx', 450)
        .attr('cy', y)
        .attr('r', r)
        .attr('fill', neg_col)
        .attr('stroke', pos_stroke)
        .attr('stroke-width', stroke_width)
    g.append('circle')
        .attr('cx', 650)
        .attr('cy', y)
        .attr('r', r)
        .attr('fill', neg_col)
        .attr('stroke', neg_stroke)
        .attr('stroke-width', stroke_width)
}

function legendText(d3, g) {
    let y = 506
    let data = [
        ['true_pos_text', 65, y, d3.format('.0f')(total * prior * true_pos) + ' true positive(s)'],
        ['false_neg_text', 265, y, d3.format('.0f')(total * prior * (1-true_pos)) + ' false negative(s)'],
        ['false_pos_text', 465, y, d3.format('.0f')(total * (1-prior) * (1-true_neg)) + ' false positive(s)'],
        ['true_neg_text', 665, y, d3.format('.0f')(total * (1-prior) * true_neg) + ' true negative(s)']
    ]
    g.selectAll('.legend_text')
        .data(data)
        .join('text')
        .attr('id', function(d, i) {
            return d[0]
        })
        .attr('class', 'legend_text')
        .attr('x', function(d, i) {
            return d[1]
        })
        .attr('y', function(d, i) {
            return d[2]
        })
        .text(function(d, i) {
            return d[3]
        })
}

function onInput(d3) {
    let q1_text = document.getElementById('q1').value
    let q2_text = document.getElementById('q2').value
    let command = 'toJS("' + q1_text + '","' + q2_text + '")'
    IPython.notebook.kernel.execute(command)
}

require.undef('viz')
define('viz', ['d3', 'slider'], function(d3, slider) {
    function draw(container) {
        d3.select(container).append('div').attr('id', 'questions')
        d3.select('#questions').append('label').attr('for', 'q1').text('What is the question you are trying to answer?').style('margin-right', '20px')
        d3.select('#questions').append('input').attr('type', 'text').attr('id', 'q1').attr('name', 'q1').attr('placeholder', 'e.g. Do I have Covid? Is Jack the killer?').style('width', '500px')
        d3.select('#q1').on('input', function() {
            onInput(d3)
        })
        d3.select('#questions').append('br')
        d3.select('#questions').append('label').attr('for', 'q2').text('What is the test you are performing?').style('margin-right', '20px')
        d3.select('#questions').append('input').attr('type', 'text').attr('id', 'q2').attr('name', 'q2').attr('placeholder', 'e.g. PCR test').style('width', '200px')

        d3.select(container).append('div').attr('id', 'sliders')
        d3.select('#sliders').append('div').attr('id', 'slider1')
        d3.select('#sliders').append('div').attr('id', 'slider2')
        d3.select('#sliders').append('div').attr('id', 'slider3')
        d3.select('#slider1').append('span').attr('class', 'slider_label').text('Prior probability').on('mouseover', function(d) {
            d3.json('base_rate.json').then(function(e) {
                let t = 'Without performing any further tests, what is the prior probability that ' + e.statement + '?'
                d3.select('.my_tooltip').text(t).style('opacity', 1)
            })
        }).on('mouseleave', function(d) {
            d3.select('.my_tooltip').style('opacity', 0)
        })
        d3.select('#slider2').append('div').attr('class', 'slider_label').text('True positive rate')
        d3.select('#slider3').append('div').attr('class', 'slider_label').text('True negative rate')
        let svg_slider1 = d3.select('#slider1').append('svg').attr('width', '280px').attr('height', '70px')
        drawSlider1(d3, slider, svg_slider1)
        let svg_slider2 = d3.select('#slider2').append('svg').attr('width', '280px').attr('height', '70px')
        drawSlider2(d3, slider, svg_slider2)
        let svg_slider3 = d3.select('#slider3').append('svg').attr('width', '280px').attr('height', '70px')
        drawSlider3(d3, slider, svg_slider3)

        d3.select(container).append('div').attr('id', 'graphic')
        d3.select('#graphic').append('svg').attr('width', graphic_width).attr('height', '600px')
        let g_circles = d3.select('#graphic svg').append('g').attr('id', 'circles')
        let g_legend = d3.select('#graphic svg').append('g').attr('id', 'legend')
        drawCircles(g_circles)
        drawLegend(g_legend)
        legendText(d3, g_legend)

        d3.select('#slider1').append('div').attr('class', 'my_tooltip')
    }
    return draw
})

element.append('Loaded 🎉')