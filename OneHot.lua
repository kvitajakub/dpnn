local OneHot, parent = torch.class('nn.OneHot', 'nn.Module')

-- adapted from https://github.com/karpathy/char-rnn
-- and https://github.com/hughperkins/char-rnn-er

function OneHot:__init(outputSize)
   parent.__init(self)
   self.outputSize = outputSize
end

local function distributeOneHot(input, output)
    if torch.type(input) == 'number' then
        if input ~= 0 then
            output[input]=1
        end
    else
        for i=1,input:size()[1] do
            distributeOneHot(input[i],output[i])
        end
    end
end

function OneHot:updateOutput(input)
   local size = input:size():totable()
   table.insert(size, self.outputSize)
   self.output:resize(unpack(size)):zero()

   distributeOneHot(input,self.output)

   return self.output
end

function OneHot:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size()):zero()
   return self.gradInput
end

function OneHot:type(type, typecache)
   self._input = nil
   return parent.type(self, type, typecache)
end
